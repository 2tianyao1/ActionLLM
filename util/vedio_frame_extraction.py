from util.opts import get_args_parser
import numpy as np
import cv2
import os
import shutil


def rename_and_collect_videos(root_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for folder_b in os.listdir(root_folder):
        path_b = os.path.join(root_folder, folder_b)

        if os.path.isdir(path_b) and folder_b.startswith('P') and folder_b[1:].isdigit():
            for folder_c in os.listdir(path_b):
                path_c = os.path.join(path_b, folder_c)

                if os.path.isdir(path_c):
                    same_vedio = []
                    for video_file in os.listdir(path_c):
                        if video_file.endswith('.avi'):
                            video_base_name, extension = os.path.splitext(video_file)

                            # split_index = video_base_name.find('_')
                            # first_part = video_base_name[:split_index]
                            # second_part = video_base_name[split_index + 1:]
                            parts = video_base_name.split('_')
                            if len(parts) > 2:
                                video_base_name = parts[0] +'_' + parts[1]
                                if video_base_name in same_vedio:
                                    continue
                                else:
                                    same_vedio.append(video_base_name)
                                    if folder_c == 'stereo':
                                        new_video_name = f"{folder_b}_{folder_c}{'01'}_{video_base_name}{extension}"
                                    else:
                                        new_video_name = f"{folder_b}_{folder_c}_{video_base_name}{extension}"

                                    source_path = os.path.join(path_c, video_file)
                                    destination_path = os.path.join(output_folder, new_video_name)

                                    shutil.copy(source_path, destination_path)
                                    print(f"Copied and renamed {video_file} to {new_video_name}")

                            else:
                                if folder_c == 'stereo':
                                    new_video_name = f"{folder_b}_{folder_c}{'01'}_{video_base_name}{extension}"
                                else:
                                    new_video_name = f"{folder_b}_{folder_c}_{video_base_name}{extension}"

                                # 构建全路径
                                source_path = os.path.join(path_c, video_file)
                                destination_path = os.path.join(output_folder, new_video_name)

                                # 复制并重命名文件
                                shutil.copy(source_path, destination_path)
                                print(f"Copied and renamed {video_file} to {new_video_name}")

def extract_frames(video_path, output_folder, num_frames):
    cap = cv2.VideoCapture(video_path)   # open vedio
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Original video frame number", total_frames)
    frame_indices = [int(total_frames / num_frames * i) for i in range(num_frames)]

    count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_filename = os.path.join(output_folder,f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            count += 1
    cap.release()
    if count != num_frames:
        is_problem = True
    else:
        is_problem = False

    return is_problem


def process_videos(args, input_folder, dataset):
    problem_vedio = []

    if dataset == 'breakfast':
        data_path = os.path.join(args.data_root,'breakfast')
        sample_rate = 6
    elif dataset == '50_salads' :
        data_path = os.path.join(args.data_root,'50_salads')
        sample_rate = 8
    features_path = os.path.join(data_path, 'features')


    for filename in os.listdir(features_path):
        if filename.endswith(".npy"):
            video_path = os.path.join(input_folder, filename.split('.')[0] + '.avi')
            feature_file = os.path.join(features_path, filename)
            features = np.load(feature_file)  # [2048,12040]
            features = features.transpose()
            print(filename, ":I3D feature number", features.shape[0])
            features = features[::sample_rate]  # extract frame
            len_features = features.shape[0]
            print("num_frames", features.shape[0])

            num_frames = len_features
            output_folder = os.path.join('/data1/ty/LLMAction_after/data/futr/vedio_extract_frame', dataset, filename.split('.')[0])

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            is_problem_vedio = extract_frames(video_path, output_folder, num_frames)

            if is_problem_vedio == True:
                problem_vedio.append(filename)

    return problem_vedio


import cv2

def extract_problem_vedio_frames(video_path, sample_rate, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    count = 0
    extracted_count = 0

    while extracted_count < num_frames and count < total_frames:
        ret, frame = cap.read()

        # 判断帧是否损坏
        if ret:
            if count % sample_rate == 0:
                frames.append(frame)
                extracted_count += 1
        count += 1

    # 如果提取的帧数不足 num_frames，从视频末尾向前提取
    if extracted_count < num_frames:
        remaining_frames_needed = num_frames - extracted_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        reverse_frames = []

        while remaining_frames_needed > 0:
            ret, frame = cap.read()
            if ret:
                reverse_frames.append(frame)
                remaining_frames_needed -= 1
            total_frames -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # 将从末尾读取的帧按顺序插入到frames列表的末尾
        frames.extend(reversed(reverse_frames))

    cap.release()
    return frames

def process_problem_videos(args, input_folder, dataset, problem_vedio):

    if dataset == 'breakfast':
        data_path = os.path.join(args.data_root,'breakfast')
        sample_rate = 6
    elif dataset == '50_salads' :
        data_path = os.path.join(args.data_root,'50_salads')
        sample_rate = 8
    features_path = os.path.join(data_path, 'features')


    for filename in problem_vedio:
        if filename.endswith(".avi"):
            video_path = os.path.join(input_folder, filename)
            feature_file = os.path.join(features_path, filename.split('.')[0] + '.npy')
            features = np.load(feature_file)  # [2048,12040]
            features = features.transpose()
            print(filename, ":I3D feature number", features.shape[0])
            features = features[::sample_rate]  # extract frame
            len_features = features.shape[0]
            print("num_frames", features.shape[0])

            num_frames = len_features
            output_folder = os.path.join('/data1/ty/LLMAction_after/data/futr/vedio_extract_frame', dataset, filename.split('.')[0])

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # is_problem_vedio = extract_frames(video_path, output_folder, num_frames)
            frames = extract_problem_vedio_frames(video_path, sample_rate, num_frames)

            # 保存提取的帧到文件或进行处理
            for i, frame in enumerate(frames):
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{i}.jpg")
                cv2.imwrite(frame_filename, frame)
                i+=1

            print(f"Total frames extracted: {len(frames)}")





def main(args):

    dataset = '50_salads'
    if dataset == '50_salads':
        vedio_data_path = '/data1/ty/LLMAction_after/data/futr/50s_video/rgb'  # 50_salads
        # problem_vedio = process_videos(args, vedio_data_path, dataset)
        problem_vedio = ['rgb-18-1.avi', 'rgb-18-2.avi', 'rgb-16-2.avi', 'rgb-17-1.avi']
        process_problem_videos(args, vedio_data_path, dataset, problem_vedio)
    if dataset == 'breakfast':
        vedio_data_path = '/data1/ty/LLMAction_after/data/futr/breakfast_vedio'  # breakfast
        new_vedio_data_path = '/data1/ty/LLMAction_after/data/futr/breakfast_vedio/bf_vedio_rename_and_collect'

        # rename_and_collect_videos(vedio_data_path, new_vedio_data_path)
        problem_vedio = process_videos(args, new_vedio_data_path, dataset)

    print(problem_vedio)




if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)