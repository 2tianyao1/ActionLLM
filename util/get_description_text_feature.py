import json

import torch
from actionllm.tokenizer import Tokenizer
from actionllm.mm_adaptation import LLMAction
from util.opts import get_args_parser
import os
from collections import OrderedDict
import numpy as np

def process_description(self, description, model_path, device):

    tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')

    action_tokens = torch.tensor(tokenizer.encode(description, bos=False, eos=False), dtype=torch.int64).to(device)   # tensor([317, 6227])
    action_tokens_emd = self.tok_embeddings(action_tokens)      # [2, 4096]
    mean_action_tokens_emd = torch.mean(action_tokens_emd, dim=0, keepdim=True)    # [1, 4096]

    one_frame_description_feature = mean_action_tokens_emd

    return one_frame_description_feature



def main(args):
    dataset = 'breakfast'
    device = torch.device(args.device)
    model_path = args.llama_model_path
    root_save_path = '/data1/ty/LLMAction_after/data/futr/vedio_extract_frame/feature_description'
    if dataset == 'breakfast':
        description_data_path = os.path.join('/data1/ty/LLMAction_after/data/futr/vedio_extract_frame/description','breakfast')
        save_path = os.path.join(root_save_path,'breakfast')
    elif dataset == '50_salads' :
        description_data_path = os.path.join('/data1/ty/LLMAction_after/data/futr/vedio_extract_frame/description','50_salads')
        save_path = os.path.join(root_save_path, '50_salads')
    model = LLMAction(args)
    model.to(device)

    for vedio in os.listdir(description_data_path):
        vedio_name = vedio.split('.')[0]
        vedio_path = os.path.join(description_data_path, vedio)
        with open(vedio_path, 'r') as f:
            vedio_frames_description = json.load(f)

        sorted_items = sorted(vedio_frames_description.items(),key=lambda item: int(os.path.splitext(item[0].split('_')[-1])[0]))
        sorted_dict = OrderedDict(sorted_items)

        one_vedio_all_frames_description_feature = []
        for frame, description in sorted_dict.items():
            one_frame_description_feature = process_description(model, description, model_path, device)
            one_vedio_all_frames_description_feature.append(one_frame_description_feature)

        combined_tensor = torch.cat(one_vedio_all_frames_description_feature, dim=0)
        cpu_combined_tensor = combined_tensor.cpu()
        np_one_vedio_feature = cpu_combined_tensor.numpy()

        file_path = os.path.join(save_path, f"{vedio_name}.npy")
        np.save(file_path, np_one_vedio_feature)
        print(f"{vedio_name}.npy has been created")









if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
