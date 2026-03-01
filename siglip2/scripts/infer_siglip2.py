"""
Usage:
    python scripts/infer_siglip2.py --model ./siglip2_runs/run1 \
                                    --image /path/to/new_bscan.png
"""
import argparse
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Fine-tuned checkpoint dir")
parser.add_argument("--image", required=True, help="Image to classify")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(args.model)
model      = SiglipForImageClassification.from_pretrained(args.model).to(device).eval()

img = Image.open(args.image).convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    pred   = logits.softmax(dim=-1).argmax(-1).item()
    label  = model.config.id2label[pred]

print(f"Prediction → {label}")
