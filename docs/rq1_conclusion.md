# RQ1 conclusion: is image harm linearly decodable?

Model: Qwen2.5-VL-7B-Instruct. Data: 750 VLSU natural images (300 unsafe, 300 topic-matched safe,
150 generic safe), each paired with a fixed harmless prompt. Final stage: probes trained on the
fit split (300 images), evaluated once on the held-out test split (300 images), commit `5d1e724`
with no uncommitted changes. Plan: [analysis_plan.md](analysis_plan.md). Raw numbers:
[results/final/](../results/final/). The pilot, used only to debug and to fix L*, is in
[results/pilot/](../results/pilot/).

## Answer

**Yes. Harmfulness in natural images is linearly decodable from Qwen2.5-VL's hidden states under
harmless text (AUROC about 0.91–0.94). But it is already decodable at the vision encoder output, so
the language model does not create it.** The signal is not explained by low-level image
statistics, the prompt wording or image topic, and it transfers to harm categories the probe never
saw.

## Pre-registered criteria (L* = layer 14, last token, chosen on the pilot)

| # | Criterion | Numbers (test AUROC, 95% bootstrap CI) | Pass |
|---|---|---|---|
| 1 | Decodable beyond the vision encoder | LR 0.908 [0.872, 0.940]; permutation 95th pct 0.557; vision-encoder baseline 0.906 [0.871, 0.939] | **no** |
| 2 | Not a low-level artefact | low-level image statistics 0.638 [0.575, 0.701] | yes |
| 3 | Text-independent | trained on prompt A, tested on B: 0.912 vs within-prompt 0.908 | yes |
| 4 | Harm, not topic | unsafe vs topic-matched safe 0.892 [0.846, 0.930]; leave-one-category-out mean 0.900 over 8 categories (range 0.806–0.951) | yes |
| 5 | A single direction | DiM 0.856 vs LR 0.908; split-half cosine 0.825; prompt A vs B cosine 0.767 | **no** |

Criterion 1 fails because of the vision-encoder baseline, not because decoding is weak.
Criterion 5 fails at the pre-specified layer only (see exploratory finding 2).

## What the results show

1. **The harm signal enters through the vision encoder.** A linear probe on the image tokens
   before any LLM layer reaches 0.906. Inside the LLM the image tokens rise slightly, to a peak of
   0.944 [0.917, 0.965] at layer 24. The CIs overlap and no paired test was run, so this gain is
   not established. The last token reaches 0.903 by layer 2. It picks up the signal from the
   image tokens almost immediately rather than building it over depth.
2. **The signal is about harm.** It survives every planned confound check:
   - Low-level statistics (size, colour, edges) reach only 0.638, and 0.591 on unsafe vs matched.
   - Image-token features cannot depend on the prompt, since the image precedes the text under
     causal attention (prompt A vs B cosine = 1.00). At the last token, cross-prompt transfer
     loses nothing.
   - Topic-matched safe images (same harm context, harmless content) are separated almost as well
     as generic safe ones (LR 0.892).
   - Leaving each harm category out of training costs little: 0.806 (C9) to 0.951 (C1).
3. **The pilot overstated the LLM's contribution.** On 150 pilot images the vision-encoder
   baseline was 0.75 and the LLM layers about 0.85. With 300 training images the baseline rises to
   0.906, which removes the gap. The pilot gap was small-sample noise.

## Exploratory findings (layers not pre-specified; not part of the verdict)

1. **A stable single direction forms later in the network.** At the last token, from about
   layer 19 on, the difference-of-means direction is close to LR in AUROC (layer 19: 0.903 vs
   0.926). It is also stable across random halves of the data (split-half cosine 0.93–0.94) and
   across prompts (A vs B cosine 0.86–0.88). Layer 14 sits just before this point. Harm is
   decodable early but settles into one consistent direction around layers 19–24.
2. **The difference-of-means direction is partly a topic direction.** At the last token it
   separates unsafe from generic safe images better than from topic-matched ones (layer 19: 0.964
   vs 0.872). LR is more robust (0.907 on matched). A probe trained only on unsafe vs matched
   reaches 0.908 (LR, layer 20).

## Caveats

- One model, one dataset. "Harm" here is VLSU's human image grade, with sexual, jailbreak and
  human-exploitation categories excluded.
- Decodability is correlational. It shows the information is linearly present, not that the
  model uses it. In an earlier VLSU behavioural baseline, Qwen refused only 1% of unsafe images
  paired with harmless text. So the model represents image harm but does not act on it by
  refusing under a harmless prompt.
- The 448x448 resize removes token-count differences but distorts aspect ratio. Aspect ratio of
  the original image is included in the low-level baseline.

## Next questions

- Is the image-harm direction the same as the text refusal direction (cosine, and probe transfer
  in both directions)?
- Does steering along it, or ablating it, change refusal behaviour?
- Is the vision encoder's harm signal linear before the merger, i.e. inside the ViT?
