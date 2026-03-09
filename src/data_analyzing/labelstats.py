import matplotlib.pyplot as plt

import os

def _get_bbox_curr_count(label_file):
    with open(label_file, 'r') as f:
        return len(f.readlines())
    
def _get_bbox_curr_rel_areas(label_file):
    areas = []
    with open(label_file, 'r') as f:
        for line in f:
            _, _, _, w, h = map(float, line.strip().split(" "))
            areas.append(w * h)
    
    return areas

def  _iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0
    
    return inter_area / union

def _get_bbox_max_ious(label_file):
    max_ious = []
    bboxes = []
    
    with open(label_file, 'r') as f:
        for line in f:
            _, x, y, w, h = map(float, line.strip().split(" "))

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            bboxes.append([x1, y1, x2, y2])
    
    for i, bbox1 in enumerate(bboxes):
        curr_max = 0
        for j, bbox2 in enumerate(bboxes):
            if i == j:
                continue
            
            curr_max = max(curr_max, _iou(bbox1, bbox2))
        
        max_ious.append(curr_max)
    
    return max_ious


def plot_label_stat(dataset_path: str, splits: list[str]):
    counts = []
    stat_name = None

    for split in splits:
        labels_dir_path = os.path.join(dataset_path, split, "labels")

        split_counts = []

        for label_file_name in os.listdir(labels_dir_path):
            label_file = os.path.join(labels_dir_path, label_file_name)

            # curr_count = _get_bbox_curr_count(label_file=label_file)
            # stat_name = "Bounding boxes per image"
            # split_counts.append(curr_count)

            # curr_count = _get_bbox_curr_rel_areas(label_file=label_file)
            # stat_name = "Bounding box relative area"
            # split_counts.extend(curr_count)

            curr_count = _get_bbox_max_ious(label_file=label_file)
            stat_name = "Max IoU with another object"
            split_counts.extend(curr_count)

        counts.extend(split_counts)
    
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=50, edgecolor="black")
    plt.title(f"Distribution of {stat_name}")
    plt.xlabel(f"{stat_name}")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.show()
def main():
    dataset_path = "dataset"
    
    splits = ["train", "valid", "test"]

    plot_label_stat(dataset_path=dataset_path, splits=splits)


if __name__ == "__main__":
    main()