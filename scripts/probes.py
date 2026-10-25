import os
import json
import argparse
import subprocess
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline

# RQ1 probes: is image harm linearly decodable from Qwen2.5-VL hidden states under harmless text?
# Implements analyses 1-7 of docs/analysis_plan.md.
#
#   --stage pilot  5-fold CV inside the pilot split. Never touches fit/test.
#   --stage final  train on fit, evaluate once on test.
#
# Every analysis is written as "train on these indices, score those", over a list of folds:
# pilot = 5 CV folds over the pilot split, final = the single (fit, test) pair. Scores are pooled
# over folds and evaluated once. Probes:
#   dim  difference of class means on raw activations, threshold at the midpoint
#   lr   standardised L2 logistic regression, C by inner 5-fold CV (fixed to that C for
#        permutations and leave-one-category-out, to keep the cost down)

PROMPTS = ["A", "B"]
POSITIONS = ["img", "last"]
CS = np.logspace(-4, 0, 5)
PERM_LR_LAYERS = [0, 7, 14, 21, 28]
MIN_LOCO = 8
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------- metrics

def auroc(y, s):
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0 or np.ptp(s) == 0:
        return 0.5
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bacc(y, s):
    pred = s > 0
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def evaluate(y, s, n_boot, rng):
    """Point estimates and stratified-bootstrap 95% CIs for AUROC and balanced accuracy."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append((auroc(y[idx], s[idx]), bacc(y[idx], s[idx])))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"auroc": auroc(y, s), "auroc_lo": lo[0], "auroc_hi": hi[0],
            "bacc": bacc(y, s), "bacc_lo": lo[1], "bacc_hi": hi[1],
            "n_pos": len(pos), "n_neg": len(neg)}


# ---------------------------------------------------------------- probes

def fit(probe, X, y, C=None):
    if probe == "dim":
        mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
        w = mu1 - mu0
        return {"w": w, "b": -0.5 * (mu1 + mu0) @ w, "C": None}
    if C is None:
        cv = LogisticRegressionCV(Cs=CS, cv=5, max_iter=3000, scoring="roc_auc")
        model = make_pipeline(StandardScaler(), cv).fit(X, y)
        C = float(cv.C_[0])
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y)
    sc, lr = model[0], model[-1]
    # Weight vector mapped back to raw activation space, for comparison with the DiM direction
    w = lr.coef_[0] / sc.scale_
    return {"w": w, "b": float(lr.intercept_[0] - (sc.mean_ * w).sum()), "C": C}


def score(p, X):
    return X @ p["w"] + p["b"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- data

class Data:
    def __init__(self, data_dir, acts_dir, lowlevel_path):
        with open(os.path.join(data_dir, "manifest.jsonl")) as f:
            man = pd.DataFrame([json.loads(line) for line in f]).set_index("uuid")
        with open(os.path.join(acts_dir, "uuids.json")) as f:
            uuids = json.load(f)
        self.man = man.loc[uuids].reset_index()
        self.y = self.man["label"].to_numpy()
        self.acts_dir = acts_dir
        self._acts = None
        self.n_layers = self.acts["A", "img"].shape[1]
        low = np.load(lowlevel_path)
        order = {u: i for i, u in enumerate(low["uuids"])}
        self.low = low["X"][[order[u] for u in uuids]]

    @property
    def acts(self):
        # Memory-mapped and opened lazily, so joblib workers don't each receive a copy
        if self._acts is None:
            self._acts = {(p, pos): np.load(os.path.join(self.acts_dir, f"{p}_{pos}.npy"), mmap_mode="r")
                          for p in PROMPTS for pos in POSITIONS}
        return self._acts

    def __getstate__(self):
        return {**self.__dict__, "_acts": None}

    def X(self, prompt, pos, layer):
        return np.asarray(self.acts[prompt, pos][:, layer], dtype=np.float32)

    def idx(self, split=None, group=None, category=None):
        m = np.ones(len(self.man), bool)
        if split is not None:
            m &= self.man["split"].isin([split] if isinstance(split, str) else split).to_numpy()
        if group is not None:
            m &= self.man["group"].isin([group] if isinstance(group, str) else group).to_numpy()
        if category is not None:
            m &= (self.man["category"] == category).to_numpy()
        return np.flatnonzero(m)


def folds(d, stage, seed):
    """(train_idx, eval_idx) pairs plus the training pool and the evaluation pool."""
    if stage == "final":
        train, ev = d.idx("fit"), d.idx("test")
        return [(train, ev)], train, ev
    pilot = d.idx("pilot")
    strat = d.man.loc[pilot, "group"].to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return [(pilot[a], pilot[b]) for a, b in skf.split(pilot, strat)], pilot, pilot


def pooled(fold_list, fit_fn, score_fn, n):
    """Fit on each fold's train indices, score its eval indices; returns full-length scores (NaN unscored)."""
    s = np.full(n, np.nan)
    for train, ev in fold_list:
        s[ev] = score_fn(fit_fn(train), ev)
    return s


# ---------------------------------------------------------------- analyses per layer

def layer_job(d, stage, layer, fold_list, train_pool, eval_pool, n_boot, n_perm_dim, n_perm_lr, seed):
    rng = np.random.default_rng(seed + layer)
    y, n = d.y, len(d.y)
    rows, dirs = [], []
    ev_idx = np.sort(np.concatenate([e for _, e in fold_list]))
    unsafe_matched = np.intersect1d(ev_idx, d.idx(group=["unsafe", "safe_matched"]))
    unsafe_generic = np.intersect1d(ev_idx, d.idx(group=["unsafe", "safe_generic"]))

    for pos in POSITIONS:
        Xs = {p: d.X(p, pos, layer) for p in PROMPTS}
        for probe in ["dim", "lr"]:
            base = {"stage": stage, "position": pos, "layer": layer, "probe": probe}
            for p_tr in PROMPTS:
                # 1, 2, 6: within- and cross-prompt, all-safe and subset evaluations from one set of fits
                fitted = {}
                for k, (train, ev) in enumerate(fold_list):
                    fitted[k] = fit(probe, Xs[p_tr][train], y[train])
                for p_ev in PROMPTS:
                    s = np.full(n, np.nan)
                    for k, (train, ev) in enumerate(fold_list):
                        s[ev] = score(fitted[k], Xs[p_ev][ev])
                    for subset, idx in [("all", ev_idx), ("unsafe_vs_matched", unsafe_matched),
                                        ("unsafe_vs_generic", unsafe_generic)]:
                        rows.append({**base, "analysis": "decode", "prompt_train": p_tr, "prompt_eval": p_ev,
                                     "subset": subset, **evaluate(y[idx], s[idx], n_boot, rng)})
                C_sel = fitted[0]["C"]

                # 6b: probe trained on unsafe vs matched safe only
                um = d.idx(group=["unsafe", "safe_matched"])
                s = pooled(fold_list, lambda tr: fit(probe, Xs[p_tr][np.intersect1d(tr, um)], y[np.intersect1d(tr, um)], C_sel),
                           lambda p, ev: score(p, Xs[p_tr][ev]), n)
                rows.append({**base, "analysis": "decode_matched_trained", "prompt_train": p_tr, "prompt_eval": p_tr,
                             "subset": "unsafe_vs_matched", **evaluate(y[unsafe_matched], s[unsafe_matched], n_boot, rng)})

                if p_tr != "A":
                    continue
                # 3: permutation null. Labels are shuffled jointly over training and evaluation images;
                # shuffling only the training labels lets a probe find the real axis with a random sign,
                # which gives a bimodal null when the signal is strong.
                n_perm = n_perm_dim if probe == "dim" else (n_perm_lr if layer in PERM_LR_LAYERS else 0)
                used = np.union1d(train_pool, eval_pool)
                null = []
                for _ in range(n_perm):
                    yp = y.copy()
                    yp[used] = rng.permutation(y[used])
                    s = np.full(n, np.nan)
                    for train, ev in fold_list:
                        s[ev] = score(fit(probe, Xs["A"][train], yp[train], C_sel), Xs["A"][ev])
                    null.append(auroc(yp[ev_idx], s[ev_idx]))
                if null:
                    rows.append({**base, "analysis": "perm_null", "prompt_train": "A", "prompt_eval": "A", "subset": "all",
                                 "null_p95": float(np.percentile(null, 95)), "null_mean": float(np.mean(null)), "n_perm": n_perm})

                # 7: leave one category out
                cats = sorted(c for c in d.man["category"].unique() if c)
                for c in cats:
                    in_c = d.idx(category=c)
                    tr = np.setdiff1d(train_pool, in_c)
                    ev = np.intersect1d(np.intersect1d(eval_pool, in_c), d.idx(group=["unsafe", "safe_matched"]))
                    if min((y[ev] == 1).sum(), (y[ev] == 0).sum()) < MIN_LOCO:
                        continue
                    p = fit(probe, Xs["A"][tr], y[tr], C_sel)
                    rows.append({**base, "analysis": "loco", "prompt_train": "A", "prompt_eval": "A", "subset": c,
                                 **evaluate(y[ev], score(p, Xs["A"][ev]), n_boot, rng)})

        # 4: direction stability (DiM on the training pool)
        Xa, Xb, yt = Xs["A"][train_pool], Xs["B"][train_pool], y[train_pool]
        w_a, w_b = fit("dim", Xa, yt)["w"], fit("dim", Xb, yt)["w"]
        halves = []
        for _ in range(20):
            perm = rng.permutation(len(yt))
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            halves.append(cos(fit("dim", Xa[h1], yt[h1])["w"], fit("dim", Xa[h2], yt[h2])["w"]))
        w_lr = fit("lr", Xa, yt)["w"]
        dirs.append({"stage": stage, "position": pos, "layer": layer,
                     "split_half_cos": float(np.mean(halves)), "split_half_cos_sd": float(np.std(halves)),
                     "cos_dim_A_B": cos(w_a, w_b), "cos_dim_lr": cos(w_a, w_lr),
                     "random_cos_sd": 1 / np.sqrt(len(w_a))})
    return rows, dirs


def lowlevel_rows(d, stage, fold_list, n_boot, seed):
    rng = np.random.default_rng(seed)
    y, X = d.y, d.low
    ev_idx = np.sort(np.concatenate([e for _, e in fold_list]))
    s = pooled(fold_list, lambda tr: fit("lr", X[tr], y[tr]), lambda p, ev: score(p, X[ev]), len(y))
    rows = []
    for subset, groups in [("all", None), ("unsafe_vs_matched", ["unsafe", "safe_matched"])]:
        idx = ev_idx if groups is None else np.intersect1d(ev_idx, d.idx(group=groups))
        rows.append({"stage": stage, "analysis": "lowlevel", "position": "lowlevel", "layer": -1, "probe": "lr",
                     "prompt_train": "-", "prompt_eval": "-", "subset": subset, **evaluate(y[idx], s[idx], n_boot, rng)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "final"], required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--acts", default="results/activations")
    parser.add_argument("--lowlevel", default="results/lowlevel.npz")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = parser.parse_args()

    out = os.path.join(args.out, args.stage)
    os.makedirs(out, exist_ok=True)
    d = Data(args.data, args.acts, args.lowlevel)
    fold_list, train_pool, eval_pool = folds(d, args.stage, args.seed)
    n_perm_dim, n_perm_lr = (200, 50) if args.stage == "final" else (50, 10)

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(layer_job)(d, args.stage, layer, fold_list, train_pool, eval_pool,
                           args.n_boot, n_perm_dim, n_perm_lr, args.seed) for layer in range(d.n_layers))
    rows = [r for rs, _ in results for r in rs] + lowlevel_rows(d, args.stage, fold_list, args.n_boot, args.seed)
    dirs = [r for _, ds in results for r in ds]
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)
    pd.DataFrame(dirs).to_csv(os.path.join(out, "directions.csv"), index=False)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "scripts", "docs"],
                                    capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        commit, dirty = "unknown", None
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump({"stage": args.stage, "commit": commit, "uncommitted_changes": dirty, "seed": args.seed,
                   "n_boot": args.n_boot, "n_perm_dim": n_perm_dim, "n_perm_lr": n_perm_lr,
                   "n_train_pool": len(train_pool), "n_eval_pool": len(eval_pool)}, f, indent=2)
    print(f"wrote {out}/metrics.csv ({len(rows)} rows), directions.csv ({len(dirs)} rows)")


if __name__ == "__main__":
    main()
