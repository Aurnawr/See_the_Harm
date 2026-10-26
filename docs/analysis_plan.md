# Analysis plan: RQ1, linear decodability of image harm

Committed before any activations are extracted. Changes after the pilot run are allowed, but
each one goes in its own commit **before** `probes.py --stage final` is run, and the final run
records the commit hash it ran on.

## Question

Can harmfulness in natural images be linearly decoded from the hidden states of
Qwen2.5-VL-7B-Instruct when the accompanying text carries no harm, and is the readout a
single stable direction rather than just some separating hyperplane?

## Data

- Source: VLSU (`apple/ml-vlsu`, `data/VLSU.csv`), images only. VLSU prompts are discarded.
- Excluded if any category field names C6, C7 (sexual), C13 (jailbreak), C15 (human exploitation).
- **unsafe**: 300 images with `image_grade == unsafe`, drawn at random from the pool.
- **safe_matched**: 300 images with `image_grade == safe` whose VLSU category (combined, else text)
  is a harm category, with per-category quotas equal to the unsafe sample's category counts.
  These are safe images from the same topical context (e.g. a kitchen knife in the weapons context).
- **safe_generic**: 150 images with `image_grade == safe` and no category.
- `borderline` / `not_sure` images are never used. Duplicate `web_path`s are dropped.
- Splits are by image, stratified on (class, group, category), seed 0:
  **pilot 20% / fit 40% / test 40%**.
- Every image is resized to **448x448** (no crop, aspect ratio not kept), which gives exactly
  256 image tokens per image. Token count therefore cannot separate the classes.

## Inputs to the model

Every image is paired with the same harmless text, under two wordings:

- prompt A: `Describe this image.`
- prompt B: `What is shown in this picture?`

Qwen chat template with its default system prompt, `add_generation_prompt=True`, no generation.

## Features

`output_hidden_states` gives 29 states (embedding output + 28 decoder layers). At each:

- `img`: mean over the 256 image-token positions
- `last`: the final prompt token (the position the first answer token is predicted from)

`img` at layer 0 is the vision encoder's output after the merger, before any LLM layer. This is
the **vision-encoder baseline**. `last` at layer 0 is a fixed template token and carries no signal.

## Probes

Labels: unsafe = 1, safe (matched + generic) = 0.

- **DiM**: difference of class means on raw activations, threshold at the midpoint of the
  projected class means.
- **LR**: standardised L2 logistic regression, C from {1e-4 ... 1} by 5-fold CV inside the
  training split.

Metrics: AUROC (primary) and balanced accuracy, with 95% CIs from 1000 stratified bootstrap
resamples of the evaluation images.

## Stages

- `pilot`: 5-fold CV inside the pilot split only. Used to debug and to check the plan. Never
  reads fit or test.
- `final`: train on fit, evaluate once on test.

## Analyses (final stage)

1. **Within-prompt decodability**: per prompt x position x layer x probe.
2. **Cross-prompt transfer**: train on A, test on B, and the reverse.
3. **Permutation null**: 200 label permutations of the training split (DiM, all layers; LR with
   the selected C, 50 permutations at layers 0, 7, 14, 21, 28). The null 95th percentile is reported.
4. **Direction stability**: split-half reliability of the DiM direction (20 random halves of the
   training split, cosine between halves); cosine between the prompt-A and prompt-B DiM
   directions; cosine between DiM and the LR weight vector mapped back to raw space.
5. **Low-level baseline**: LR on image statistics from the original files (width, height,
   aspect ratio, file size, RGB mean/std, HSV saturation/value, edge density, near-white/black
   fraction, colourfulness).
6. **Topic-matched contrast**: the main probe evaluated on test restricted to unsafe vs
   safe_matched, and separately unsafe vs safe_generic. A second LR is trained on unsafe vs
   safe_matched only.
7. **Leave-one-category-out**: for each harm category c, train on the fit split excluding all
   images of category c (unsafe and matched safe), evaluate on category-c images from fit + test
   (unsafe c vs safe_matched c). Reported only where both sides have >= 8 images.

## Decision criteria (review after the pilot, then freeze)

Let L* be the layer with the highest LR AUROC at the `last` position under prompt A, **chosen on
the pilot stage** so the test split plays no part in picking it. Pilot result: **L* = 14**.
All layers are still reported.

(Amended after the pilot. The original rule picked L* on the test split, which inflates the
estimate. No other criterion was changed.)

- **Decodable from the LLM, beyond the vision encoder**: LR AUROC CI lower bound at L* > 0.75,
  above the permutation 95th percentile, and > (vision-encoder baseline AUROC + 0.05).
- **Not a low-level artefact**: the low-level baseline AUROC is below 0.65 or at least 0.10 below L*.
- **Text-independent**: cross-prompt AUROC at L* is within 0.05 of within-prompt AUROC.
- **Harm, not topic**: unsafe-vs-matched AUROC at L* >= 0.70, and mean LOCO AUROC >= 0.70.
- **A single direction**: DiM AUROC within 0.05 of LR at L*, split-half cosine >= 0.8, and
  prompt-A vs prompt-B DiM cosine >= 0.8.

Each criterion is reported as pass/fail with its numbers. No overall verdict is computed by code.
