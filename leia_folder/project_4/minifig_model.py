# Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export, class "Minifig", 49 images).
# Training progress and the final weights land under
# runs/detect/green_minifig/weights/best.pt

from pathlib import Path

from ultralytics import YOLO


DATA_YAML = Path(__file__).parent.parent.parent.parent.parent / "lego_data" / "data.yaml"
EPOCHS = 100  # small dataset (49 images) benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = 0  # NVIDIA GPU

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with a few dozen images.
model = YOLO("yolov8n.pt")

try:
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        device=DEVICE,
    )
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        device="cpu",
    )

print("Done. Weights saved to runs/detect/green_minifig/weights/best.pt")
