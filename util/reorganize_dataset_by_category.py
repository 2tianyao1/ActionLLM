import os
import shutil
import json

# 定义根文件夹路径
root_folder = '/data1/ty/LLMAction_after/data/futr/resnet_50_vedio_extract_frame/50_salads'


# 创建一个函数来处理每个分文件夹
def process_folder(folder_path,video_name, dataset):
    # 获取该文件夹下的所有文件
    files = os.listdir(folder_path)

    json_file_path = '/data1/ty/LLMAction_after/data/futr/resnet_50_vedio_extract_frame/frame_label'

    # 读取.json文件中的标签列表
    with open(os.path.join(json_file_path, dataset, video_name+'.json'), 'r') as f:
        labels = json.load(f)

    # 创建一个字典来存储标签到图片名的映射
    label_to_images = {}

    # 遍历图片文件并根据标签分类
    for i, label in enumerate(labels):
        image_name = f"{video_name}_frame_{i}.jpg"
        image_path = os.path.join(folder_path, image_name)

        if not os.path.isfile(image_path):
            print(f"Image file {image_name} not found in {folder_path}")
            continue

        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(image_path)

    # 将图片移动到相应的标签文件夹中
    for label, images in label_to_images.items():
        label_folder = os.path.join('/data1/ty/LLMAction_after/data/futr/resnet_50_vedio_extract_frame/reorganize_dataset_by_category', dataset, label)
        os.makedirs(label_folder, exist_ok=True)

        for image_path in images:
            shutil.copy2(image_path, os.path.join(label_folder, os.path.basename(image_path)))


# 遍历总文件夹a中的所有分文件夹b
for subfolder in os.listdir(root_folder):
    dataset = '50_salads'
    subfolder_path = os.path.join(root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        process_folder(subfolder_path,subfolder,dataset)
