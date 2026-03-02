from ultralytics import YOLO
import torch

from datetime import datetime
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'yolo11n_train_{timestamp}'

    model = YOLO("yolo11n.pt")

    print("Starting training...")
    model.train(
        data=os.path.join("dataset", "data.yaml"),
        epochs=25,
        imgsz=640,
        save=True,
        val=True,
        plots=True,
        batch=16,
        device=device,
        patience=10,
        workers=4,
        project='runs/plantdoc',
        name=run_name
    )


if __name__ == "__main__":
    main()