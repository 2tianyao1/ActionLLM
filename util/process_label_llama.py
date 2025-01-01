import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from util.opts import get_args_parser
from actionllm import Tokenizer



def seq2list(self, action_dict):
    transcript_action = []  # 存储转录动作
    for key in action_dict.keys():
        transcript_action.append(key)

    return transcript_action

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def process_all_class_list(all_class_list, model_path, device):
    vectors = []
    tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
    for action in all_class_list:     # SLI
        action_tokens = torch.tensor(tokenizer.encode(action, bos=False, eos=False), dtype=torch.int64).to(device)   # tensor([317, 6227])
        action_tokens_emd = self.tok_embeddings(action_tokens)      # [2, 4096]
        mean_action_tokens_emd = torch.mean(action_tokens_emd, dim=0, keepdim=True)    # [1, 4096]
        vectors.append(mean_action_tokens_emd)

    all_class_vectors = torch.cat(vectors)

    return all_class_vectors

def main(args):
    device = torch.device(args.device)

    # dataset = 'breakfast'
    # dataset= '50_salads'

    tokenizer = Tokenizer(model_path=args.llama_model_path + '/tokenizer.model')

    # if dataset == 'breakfast':
    #     data_path = os.path.join(args.data_root,'breakfast')
    # elif dataset == '50_salads' :
    #     data_path = os.path.join(args.data_root,'50_salads')
    #
    # gt_path = os.path.join(data_path, 'groundTruth')
    # mapping_file = os.path.join(data_path, 'mapping.txt')
    # actions_dict = read_mapping_dict(mapping_file)
    # all_label_list = seq2list(actions_dict)


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)



