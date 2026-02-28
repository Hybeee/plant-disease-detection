import matplotlib.pyplot as plt
import numpy as np

import os
import yaml

def get_class_indices(file_path):
    class_indices = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_indices.append(int(parts[0]))
    
    return class_indices

def plot_label_distribution(split, class_names, class_counts):
    max_count = max(class_counts)
    total = sum(class_counts)

    x_labels = class_names

    plt.figure(figsize=(12, 8))
    bars = plt.barh(x_labels, class_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title(f"PlantDoc {split} Distribution")
    plt.xlabel("Number of Samples")
    plt.ylabel("Class Labels")
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + max_count * 0.01,
                 bar.get_y() + bar.get_height()/2,
                 f'{int(width)}',
                 va='center')


    plt.tight_layout()
    plt.show()

def main():
    with open(os.path.join("dataset", "data.yaml")) as f:
        data_info = yaml.safe_load(f)

    class_names = np.array(data_info["names"])
    class_counts = np.zeros(class_names.shape)

    split = "train"

    labels_dir = os.path.join("dataset", split, "labels")

    for label_file in os.listdir(labels_dir):
        class_indices = get_class_indices(file_path=os.path.join(labels_dir, label_file))
        np.add.at(class_counts, class_indices, 1)

    plot_label_distribution(split=split, class_names=class_names, class_counts=class_counts)

if __name__ == "__main__":
    main()