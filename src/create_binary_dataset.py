import os
import yaml
import shutil

def _get_relevant_dirs(root, split, create=False):
    dir = os.path.join(root, split)
    dir_images = os.path.join(dir, "images")
    dir_labels = os.path.join(dir, "labels")

    if create:
        os.makedirs(dir, exist_ok=True)
        os.makedirs(dir_images, exist_ok=True)
        os.makedirs(dir_labels, exist_ok=True)

    return dir_images, dir_labels


def _create_new_split_dir(orig_root: str, new_root: str, split: str,
                          leaf_health_status:dict, orig_data_info: dict):
    orig_images_dir, orig_old_labels_dir = _get_relevant_dirs(root=orig_root, split=split)

    new_images_dir, new_labels_dir = _get_relevant_dirs(root=new_root, split=split, create=True)

    for orig_image_file_name, orig_label_file_name in zip(sorted(os.listdir(orig_images_dir)), sorted(os.listdir(orig_old_labels_dir))):
        orig_image_path = os.path.join(orig_images_dir, orig_image_file_name)
        new_image_path = os.path.join(new_images_dir, orig_image_file_name)
        shutil.copy2(orig_image_path, new_image_path)

        orig_label_path = os.path.join(orig_old_labels_dir, orig_label_file_name)
        new_label_path = os.path.join(new_labels_dir, orig_label_file_name)

        with open(orig_label_path, 'r') as orig_f, open(new_label_path, 'w') as new_f:
            for line in orig_f:
                parts = line.strip().split()

                orig_class_id = int(parts[0])
                orig_class_name = orig_data_info["names"][orig_class_id]
                new_class_id = leaf_health_status[orig_class_name]

                new_line = " ".join([str(new_class_id)] + parts[1:])
                new_f.write(f"{new_line}\n")

def main():
    leaf_health_status = {
        "Apple Scab Leaf": 1,
        "Apple leaf": 0,
        "Apple rust leaf": 1,
        "Bell_pepper leaf": 0,
        "Bell_pepper leaf spot": 1,
        "Blueberry leaf": 0,
        "Cherry leaf": 0,
        "Corn Gray leaf spot": 1,
        "Corn leaf blight": 1,
        "Corn rust leaf": 1,
        "Peach leaf": 0,
        "Potato leaf": 0,
        "Potato leaf early blight": 1,
        "Potato leaf late blight": 1,
        "Raspberry leaf": 0,
        "Soyabean leaf": 0,
        "Soybean leaf": 0,
        "Squash Powdery mildew leaf": 1,
        "Strawberry leaf": 0,
        "Tomato Early blight leaf": 1,
        "Tomato Septoria leaf spot": 1,
        "Tomato leaf": 0,
        "Tomato leaf bacterial spot": 1,
        "Tomato leaf late blight": 1,
        "Tomato leaf mosaic virus": 1,
        "Tomato leaf yellow virus": 1,
        "Tomato mold leaf": 1,
        "Tomato two spotted spider mites leaf": 1,
        "grape leaf": 0,
        "grape leaf black rot": 1
    }

    orig_dataset_dir = "dataset"
    with open(os.path.join(orig_dataset_dir, "data.yaml")) as f:
        orig_data_info = yaml.safe_load(f)
    
    new_dataset_dir = "binary_dataset"
    os.makedirs(new_dataset_dir, exist_ok=True)

    splits = ["train", "valid", "test"]

    for split in splits:
        _create_new_split_dir(
            orig_root=orig_dataset_dir,
            new_root=new_dataset_dir,
            split=split,
            leaf_health_status=leaf_health_status,
            orig_data_info=orig_data_info
        )

    new_data_info = orig_data_info.copy()
    new_data_info["nc"] = 2
    new_data_info["names"] = ["Healthy", "Diseased"]

    with open(os.path.join(new_dataset_dir, "data.yaml"), 'w') as f:
        yaml.safe_dump(new_data_info, f, sort_keys=False)

if __name__ == "__main__":
    main()