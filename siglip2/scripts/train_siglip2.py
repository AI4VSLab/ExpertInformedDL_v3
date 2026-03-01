"""
Fine-tune SigLIP 2 for 2-class image classification (AMD vs Normal).

Run:
    python scripts/train_siglip2.py \
        --pickle ~/ExpertInformedDL_v3/bscan_imgs.p \
        --out_dir ./siglip2_runs/run1
"""
import argparse, os, pickle, gc, warnings, random

import numpy as np
from PIL import Image
from datasets import Dataset, ClassLabel
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip, RandomRotation,
    ToTensor, Normalize, Resize
)

# from transformers import (
#     AutoImageProcessor, SiglipForImageClassification,
#     TrainingArguments, Trainer, DefaultDataCollator
# )
from transformers import (
    AutoImageProcessor, SiglipForImageClassification,
    Trainer, DefaultDataCollator
)
from transformers import TrainingArguments
import evaluate

warnings.filterwarnings("ignore")
RNG = 42

# --------------------------------------------------------------------- #
# 1. CLI
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--pickle", required=True, help="Path to bscan_imgs.p")
parser.add_argument("--out_dir", default="./siglip2_runs/run1", help="Where checkpoints & logs go")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch",  type=int, default=8)
parser.add_argument("--lr",     type=float, default=5e-5)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(RNG); np.random.seed(RNG); torch.manual_seed(RNG)

# --------------------------------------------------------------------- #
# 2.  Load pickle → 🤗  Dataset
# --------------------------------------------------------------------- #
print("→ Loading pickle …")
data_pkl = pickle.load(open(args.pickle, "rb"))

images, labels = [], []
for meta in data_pkl.values():
    img = Image.fromarray(meta["original_image"]).convert("RGB")
    lab = 0 if meta["label"] == "N" else 1   # 0 = Normal, 1 = AMD
    images.append(img); labels.append(lab)

print("→ Creating Dataset …")

dataset = Dataset.from_dict({"image": images, "label": labels})
# ── after we create `dataset` ─────────────────────────────────────────────
dataset = dataset.cast_column(
    "label",
    ClassLabel(num_classes=2, names=["Normal", "AMD"])
)   # NEW ← transforms int → ClassLabel

train_ds, valid_ds = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="label",
    seed=RNG
).values()
# ─────────────────────────────────────────────────────────────────────────

# train_ds, valid_ds = dataset.train_test_split(test_size=0.2,
#                                               stratify_by_column="label",
#                                               seed=RNG).values()

print(f"→ Train dataset: {len(train_ds)} samples")
print(f"→ Valid dataset: {len(valid_ds)} samples")
# --------------------------------------------------------------------- #
# 3.  Image processor & transforms
# --------------------------------------------------------------------- #
ckpt_name = "google/siglip2-base-patch16-224"
processor  = AutoImageProcessor.from_pretrained(ckpt_name)
mean, std  = processor.image_mean, processor.image_std
size       = processor.size["height"]      # 224

train_tfm = Compose([
    RandomResizedCrop(size, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize(mean, std)
])
val_tfm = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean, std)
])

def set_transform(ds, tfm):
    return ds.with_transform(lambda ex: {
        "pixel_values": [tfm(img) for img in ex["image"]],
        "labels":       ex["label"]
    })

train_ds = set_transform(train_ds, train_tfm)
valid_ds = set_transform(valid_ds, val_tfm)

# --------------------------------------------------------------------- #
# 4.  Model
# --------------------------------------------------------------------- #
id2label = {0: "Normal", 1: "AMD"}
label2id = {v: k for k, v in id2label.items()}

print("→ Initializing model …")
model = SiglipForImageClassification.from_pretrained(
    ckpt_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# --------------------------------------------------------------------- #
# 5.  Metrics
# --------------------------------------------------------------------- #
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = logits.argmax(-1)
    return {"accuracy": accuracy.compute(predictions=y_pred, references=y_true)["accuracy"]}

# --------------------------------------------------------------------- #
# 6.  Trainer
# --------------------------------------------------------------------- #
print("→ Initializing Trainer …")
args_tr = TrainingArguments(
    output_dir=args.out_dir,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=RNG,
)

trainer = Trainer(
    model=model,
    args=args_tr,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DefaultDataCollator(return_tensors="pt"),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(args.out_dir)          # final best checkpoint

print("✓ Training done. Best model saved to", args.out_dir)
