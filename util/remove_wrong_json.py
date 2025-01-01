import os

# 假设 a 文件夹的路径
a_path = "/data1/ty/LLMAction_after/data/futr/vedio_extract_frame/breakfast"

# 遍历 a 文件夹中的所有子文件夹 b
for b_folder_name in os.listdir(a_path):
    b_folder_path = os.path.join(a_path, b_folder_name)

    # 检查 b 是不是文件夹
    if os.path.isdir(b_folder_path):

        # 遍历 b 文件夹中的所有文件
        for file_name in os.listdir(b_folder_path):
            file_path = os.path.join(b_folder_path, file_name)

            # 检查文件是否是 .json 文件
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                os.remove(file_path)
                print(f"Removed: {file_path}")

print("All .json files have been removed from each b folder.")