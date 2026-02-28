import cv2
import matplotlib.pyplot as plt

import os
import yaml

def return_folder_content(root):
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")

    return os.listdir(images_dir), os.listdir(labels_dir)

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
    with open(os.path.join("dataset", "data.yaml")) as f:
        info = yaml.safe_load(f)
    
    class_names = info["names"]

    train_dir_path = os.path.join("dataset", "train")
    valid_dir_path = os.path.join("dataset", "valid")
    test_dir_path = os.path.join("dataset", "test")

    train_image_names, train_label_names = return_folder_content(root=train_dir_path)
    valid_image_names, valid_label_names = return_folder_content(root=valid_dir_path)
    test_image_names, test_label_names = return_folder_content(root=test_dir_path)

    train_length = len(os.listdir(os.path.join(train_dir_path, "images")))
    valid_length = len(os.listdir(os.path.join(valid_dir_path, "images")))
    test_length = len(os.listdir(os.path.join(test_dir_path, "images")))

    while True:
        split_option = input("Enter train/valid/test to view images in the respective split or exit... ")

        if split_option.lower() == "train":
            split_dir = train_dir_path
            split_image_names = train_image_names
            split_label_names = train_label_names
            index = int(input(f"Enter a number between 0-{train_length}: "))
        elif split_option.lower() == "valid":
            split_dir = valid_dir_path
            split_image_names = valid_image_names
            split_label_names = valid_label_names
            index = int(input(f"Enter a number between 0-{valid_length}: "))
        elif split_option.lower() == "test":
            split_dir = test_dir_path
            split_image_names = test_image_names
            split_label_names = test_label_names
            index = int(input(f"Enter a number between 0-{test_length}: "))
        elif split_option.lower() == "exit":
            break
        else:
            print("Invalid split.")
            continue

        image = cv2.imread(os.path.join(split_dir, "images", split_image_names[index]))
        classes, bboxes = get_labels(labels_path=os.path.join(split_dir, "labels", split_label_names[index]))
        
        image = draw_info_on_image(image=image,
                                   classes=classes,
                                   bboxes=bboxes,
                                   class_names=class_names)
        
        cv2.imshow("image", image)
        cv2.waitKey()
        

if __name__ == "__main__":
    main()