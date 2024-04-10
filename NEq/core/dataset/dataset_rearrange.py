import os
import shutil
from sklearn.model_selection import train_test_split

def rearrange_pet_dataset(dataset_root, output_root, validation_split=0.2, random_seed=42):
    # Create output folders
    train_output_folder = os.path.join(output_root, 'train')
    val_output_folder = os.path.join(output_root, 'val')
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(val_output_folder, exist_ok=True)

    # Get the list of subdirectories (pet classes) in the original dataset
    classes = [d for d in os.listdir(os.path.join(dataset_root, 'images')) if os.path.isdir(os.path.join(dataset_root, 'images', d))]

    # Split classes into training and validation sets
    train_classes, val_classes = train_test_split(classes, test_size=validation_split, random_state=random_seed)

    # Move images to the new structure
    for class_name in classes:
        class_folder = os.path.join(dataset_root, 'images', class_name)
        images = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image in images:
            source_path = os.path.join(class_folder, image)

            if class_name in train_classes:
                destination_folder = os.path.join(train_output_folder, class_name)
            else:
                destination_folder = os.path.join(val_output_folder, class_name)

            os.makedirs(destination_folder, exist_ok=True)
            destination_path = os.path.join(destination_folder, image)

            shutil.copy(source_path, destination_path)

if __name__ == "__main__":
    # Set the paths to your dataset and desired output
    dataset_root = "/home/vanchien/on_device_learning/data/datasets/pets37"  # Change this to the actual path
    output_root = "/home/vanchien/on_device_learning/data/datasets/pets37"  # Change this to the desired output path

    # Rearrange the dataset
    rearrange_pet_dataset(dataset_root, output_root)
