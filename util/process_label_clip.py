import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from util.opts import get_args_parser
import numpy as np
import clip




def main(args):
    device = torch.device(args.device)

    dataset = 'breakfast'
    # dataset= '50_salads'
    split = args.split
    clipmodel = clip.load('ViT-L/14', device=device)[0]

    if dataset == 'breakfast':
        data_path = os.path.join(args.data_root,'breakfast')
    elif dataset == '50_salads' :
        data_path = os.path.join(args.data_root,'50_salads')

    gt_path = os.path.join(data_path, 'groundTruth')
    # path after processing text labels with clip encoding
    gt_vector_path= os.path.join(data_path, 'groundTruth_vector')

    mapping_file = os.path.join(data_path, 'mapping.txt')
    actions_dict = read_mapping_dict(mapping_file)
    action_dict_keys = list(actions_dict.keys())

    input_folder = gt_path
    output_folder = gt_vector_path

    label_map_vector_dict = {}
    for string in action_dict_keys:

        label_token = clip.tokenize(string).to(device)
        label_feature = clipmodel.encode_text(label_token).to(device)

        label_map_vector_dict[string] = label_feature


    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            # Convert text to vector with clip
            vectors = []
            with open(file_path, "r") as file:
                for line in file:
                    line = line.rstrip("\n")
                    if line in label_map_vector_dict:
                        vector = label_map_vector_dict[line]
                        vectors.append(vector)


            matrix = torch.cat(vectors)
            matrix = matrix.to('cpu')
            matrix=matrix.detach().numpy()
            output_file = os.path.join(output_folder, filename[:-4])
            if not os.path.exists(output_file):

                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

            # save as .npy
            np.save(output_file, matrix)

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)