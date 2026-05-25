from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import os
import yaml

def get_labels(labels_path):
    classes, bboxes = [], []

    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            
            class_id = int(parts[0])
            bbox = tuple(map(float, parts[1:]))

            classes.append(class_id)
            bboxes.append(bbox)
    
    return classes, bboxes

def draw_info_on_image(image, classes, bboxes, class_names):
    h, w = image.shape[:2]

    for class_id, bbox in zip(classes, bboxes):
        x_rel_c, y_rel_c, w_rel, h_rel = bbox

        x_c = x_rel_c * w
        y_c = y_rel_c * h
        bw = w * w_rel
        bh = h * h_rel

        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

        text = class_names[class_id]
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        text_x = x1
        text_y = y1 - 5
        
        if text_y - text_h < 0:
            text_y = y1 + text_h + 5
        if text_x + text_w > w:
            text_x = w - text_w
        
        text_x = max(0, text_x)
        text_y = max(text_h, text_y)

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1
        )
    
    return image

def main():
    dataset_dir = "binary_dataset"

    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        data_info = yaml.safe_load(f)

    images_dir = "binary_dataset/train/images"
    labels_dir = "binary_dataset/train/labels"

    model = YOLO("runs/detect/runs/plantdoc_binary/yolo11n_train_20260304_230941/weights/best.pt")

    save_dir = "train_inference"
    os.makedirs(save_dir, exist_ok=True)

    for image_name in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        image_name_we = os.path.splitext(os.path.basename(image_path))[0]

        labels_path = os.path.join(labels_dir, image_name_we) + ".txt"
        classes, bboxes = get_labels(labels_path)
        
        image = cv2.imread(image_path)
        gt_image = draw_info_on_image(image, classes, bboxes, data_info["names"])
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        results =  model(image_path)
        result = results[0]

        result_image = result.plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs = axs.flatten()
        for ax in axs:
            ax.axis('off')

        axs[0].imshow(gt_image)
        axs[0].set_title("Ground Truth")

        axs[1].imshow(result_image)
        axs[1].set_title("Prediction")

        plt.tight_layout()

        save_path = os.path.join(save_dir, image_name)
        plt.savefig(save_path)
        plt.close()

        print(f"Saved comparison: {save_path}")

if __name__ == "__main__":
    main()