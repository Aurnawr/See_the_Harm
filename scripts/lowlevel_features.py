import os
import json
import argparse
import numpy as np
from PIL import Image

# Low-level image statistics for the confound baseline, computed on the original downloaded files
# (before the 448x448 resize), so they also capture size and aspect ratio. CPU only.
# Output: results/lowlevel.npz with X [N, F], feature names, and uuids in manifest order.

FEATURES = ["width", "height", "log_aspect", "log_pixels", "log_file_bytes",
            "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
            "sat_mean", "val_mean", "edge_density", "frac_white", "frac_black", "colourfulness"]


def features(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    gray = rgb.mean(-1)
    gy, gx = np.gradient(gray)
    edge = float((np.hypot(gx, gy) > 0.1).mean())
    # Hasler & Suesstrunk colourfulness
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colourful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    return [w, h, np.log(w / h), np.log(w * h), np.log(os.path.getsize(path)),
            *rgb.mean((0, 1)), *rgb.std((0, 1)), hsv[..., 1].mean(), hsv[..., 2].mean(), edge,
            float((gray > 0.95).mean()), float((gray < 0.05).mean()), colourful]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/lowlevel.npz")
    args = parser.parse_args()
    with open(os.path.join(args.data, "manifest.jsonl")) as f:
        items = [json.loads(line) for line in f]
    X = np.array([features(os.path.join(args.data, it["image"])) for it in items], dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, names=np.array(FEATURES), uuids=np.array([it["uuid"] for it in items]))
    print(f"{X.shape[0]} images, {X.shape[1]} features -> {args.out}")


if __name__ == "__main__":
    main()
