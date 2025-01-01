import numpy as np
import torch
import torch.nn as nn
import os
import pdb
import torch.nn.functional as F
import datetime
import shutil



def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def backup_code(code_version):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M")
    res_dir = './result/code/' + code_version +time_str + '/'
    os.makedirs(res_dir)
    # log_file = res_dir + 'results.txt'

    file_list = [ 'engine.py' ,'opts.py' ,'eval.py','train.py','setup.py']
    fol_list = ['clip','data','actionllm','util','scripts']

    for file in file_list:
        shutil.copyfile(file, res_dir + file)
    for folname in fol_list:
        shutil.copytree(folname, res_dir + folname)
    print('code backup at %s' % res_dir)
    return


def normalize_duration(input, mask):
    if mask is not None:
        input = torch.exp(input)*mask
        output = F.normalize(input, p=1, dim=-1)
    else:
        input = torch.exp(input)
        output = F.normalize(input, p=1, dim=-1)
    return output

