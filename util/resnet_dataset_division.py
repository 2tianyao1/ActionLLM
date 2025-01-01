import os
import shutil
import random


def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    categories = os.listdir(input_dir)
    for category in categories:
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # Create directories for train, val and test
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)

        # Copy images to respective directories
        for image in train_images:
            shutil.copy2(os.path.join(category_path, image), os.path.join(output_dir, 'train', category, image))

        for image in val_images:
            shutil.copy2(os.path.join(category_path, image), os.path.join(output_dir, 'val', category, image))

        for image in test_images:
            shutil.copy2(os.path.join(category_path, image), os.path.join(output_dir, 'test', category, image))

    print("Dataset split completed.")


# 使用方法
input_dir = '/data1/ty/LLMAction_after/data/futr/resnet_50_vedio_extract_frame/reorganize_dataset_by_category/50_salads'  # 原始数据集路径
output_dir = '/data1/ty/LLMAction_after/data/futr/resnet_50_vedio_extract_frame/reorganize_dataset_by_category/split_training_and_testing/50_salads'  # 划分后数据集的存放路径

split_dataset(input_dir, output_dir)
