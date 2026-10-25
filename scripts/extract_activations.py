import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Caches Qwen2.5-VL-7B-Instruct hidden states for every image in data/manifest.jsonl under each
# neutral prompt. GPU job (slurm/extract.sbatch).
#
# Each image is resized to 448x448 and the processor is pinned to exactly that many pixels, so every
# image gives 256 image tokens. output_hidden_states gives 29 states (embedding output + 28 layers);
# at each we keep
#   img   mean over the image-token positions   (layer 0 = vision encoder output after the merger)
#   last  the final prompt token                (where the first answer token is predicted from)
# Output per prompt: results/activations/<prompt>_{img,last}.npy, float16 [N, 29, 3584], rows in
# the order of uuids.json.

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPTS = {"A": "Describe this image.", "B": "What is shown in this picture?"}
SIDE = 448
N_IMAGE_TOKENS = (SIDE // 28) ** 2


def load_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.jsonl")) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def states(model, processor, image, prompt, device, image_token_id):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    img_pos = inputs["input_ids"][0] == image_token_id
    assert int(img_pos.sum()) == N_IMAGE_TOKENS, f"expected {N_IMAGE_TOKENS} image tokens, got {int(img_pos.sum())}"
    hs = model(**inputs, output_hidden_states=True).hidden_states
    img = torch.stack([h[0, img_pos].float().mean(0) for h in hs])
    last = torch.stack([h[0, -1].float() for h in hs])
    return img.cpu().numpy().astype(np.float16), last.cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/activations")
    parser.add_argument("--limit", type=int, default=None, help="first N images only, for a smoke test")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    image_token_id = model.config.image_token_id

    items = load_manifest(args.data)[:args.limit]
    with open(os.path.join(args.out, "uuids.json"), "w") as f:
        json.dump([it["uuid"] for it in items], f)

    for key, prompt in PROMPTS.items():
        img_all, last_all = [], []
        for i, it in enumerate(items):
            image = Image.open(os.path.join(args.data, it["image"])).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
            img, last = states(model, processor, image, prompt, device, image_token_id)
            img_all.append(img)
            last_all.append(last)
            if (i + 1) % 50 == 0:
                print(f"prompt {key}: {i + 1}/{len(items)}", flush=True)
        np.save(os.path.join(args.out, f"{key}_img.npy"), np.stack(img_all))
        np.save(os.path.join(args.out, f"{key}_last.npy"), np.stack(last_all))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"model": MODEL_ID, "prompts": PROMPTS, "side": SIDE, "n_image_tokens": N_IMAGE_TOKENS,
                   "n": len(items), "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__}, f, indent=2)


if __name__ == "__main__":
    main()
