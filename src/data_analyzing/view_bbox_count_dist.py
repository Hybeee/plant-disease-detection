import matplotlib.pyplot as plt

import os

def main():
    dataset_path = "dataset"
    
    splits = ["train", "valid", "test"]

    counts = []

    for split in splits:
        labels_dir_path = os.path.join(dataset_path, split, "labels")
        
        split_counts = []

        for label_file_name in os.listdir(labels_dir_path):
            label_file = os.path.join(labels_dir_path, label_file_name)

            with open(label_file, 'r') as f:
                curr_count = len(f.readlines())
                split_counts.append(curr_count)
        
        counts.extend(split_counts)

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=50, edgecolor="black")
    plt.title("Distribution of number of bounding boxes on images")
    plt.xlabel("Number of bounding boxes per image")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()