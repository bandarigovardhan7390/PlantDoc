
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_tomato_model.pth"
OUTPUT_DIR = PROJECT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

ONNX_PATH = OUTPUT_DIR / "tomato_mobilenetv3.onnx"
CLASSES_PATH = OUTPUT_DIR / "class_names.json"

# Load the saved checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

if len(class_names) != 10 or image_size != 224:
    raise ValueError("Unexpected model metadata")

# Rebuild exactly the training architecture
model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Fixed batch size of 1 for mobile inference
dummy_input = torch.randn(1, 3, image_size, image_size)

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamo=False
)

# Preserve the exact class order
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

# Validate the exported ONNX model
import onnx
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX export successful")
print("Model:", ONNX_PATH)
print("Classes:", CLASSES_PATH)
print("Input shape:", [1, 3, 224, 224])
print("Output classes:", len(class_names))
print("Recorded validation accuracy:",
      checkpoint["best_val_accuracy"] * 100, "%")
