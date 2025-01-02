from typing import Iterable
import json

import numpy as np
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import os
import torch.nn.functional as F
import cv2
from util.action_tool import normalize_duration
import copy
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (item) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        inputs_embeds, text_inputs_embeds, past_labels, labels_action , labels_duration = item

        inputs_embeds = inputs_embeds.to(device)
        text_inputs_embeds = text_inputs_embeds.to(device)
        past_labels = past_labels.to(device)
        labels_duration = labels_duration.to(device)
        labels_action = labels_action.to(device)

        loss = model(inputs_embeds=inputs_embeds, text_inputs_embeds=text_inputs_embeds, past_labels=past_labels, labels_duration=labels_duration, labels_action=labels_action)
        loss_value = loss.item()

        if torch.isnan(loss):
            print("NaN loss encountered. Skipping this batch.")
            continue
        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # todo: improve here
        torch.cuda.synchronize()

        metric_logger.update(closs=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_file(gt_content, recog_content, obs_percentage, classes):
    # github.com/yabufarha/anticipating-activities
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[classes[ground_truth[i]]] += 1
        else:
            n_F[classes[ground_truth[i]]] += 1

    return n_T, n_F

def eval_file_all_none(gt_content, obs_percentage, classes, p):
    # github.com/yabufarha/anticipating-activities

    ground_truth = gt_content[int(obs_percentage * len(gt_content)):int((obs_percentage+p) * len(gt_content))]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        n_F[classes[ground_truth[i]]] += 1

    return n_T, n_F


def seq2idx(seq, action_dict):
    idx = np.zeros(len(seq))
    for i in range(len(seq)):
        idx[i] = action_dict[seq[i]]
    return idx


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    all_loss = list()
    for item in data_loader:
        inputs_embeds, text_inputs_embeds, past_labels, labels_action, labels_duration = item

        inputs_embeds = inputs_embeds.to(device)
        text_inputs_embeds = text_inputs_embeds.to(device)
        past_labels = past_labels.to(device)
        labels_duration = labels_duration.to(device)
        labels_action = labels_action.to(device)

        with torch.no_grad():
            loss = model(inputs_embeds=inputs_embeds, text_inputs_embeds=text_inputs_embeds, past_labels=past_labels,labels_duration=labels_duration, labels_action=labels_action)
        loss_item = loss.item()
        all_loss.append(loss_item)

    all_loss_np = np.asarray(all_loss)

    return all_loss_np


def predict(model, vid_list, args, obs_p, n_class, actions_dict, device, data_path,res_des,text_feature_path):
    model.eval()
    with torch.no_grad():
        features_path = os.path.join(data_path, 'features')
        gt_path = os.path.join(data_path, 'groundTruth')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        for vid in vid_list:
            file_name = vid.split('/')[-1].split('.')[0]

            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_seq = gt_seq[::sample_rate]  # extract frame
            gt_read.close()

            # load features
            features_file = os.path.join(features_path, file_name+'.npy')
            features = np.load(features_file).transpose()
            features = features[::sample_rate]

            text_feature_file = os.path.join(text_feature_path, file_name + '.npy')
            text_features = np.load(text_feature_file)  # [2048,12040]

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len)
            future_len = int(pred_p*vid_len)

            past_seq = gt_seq[:past_len]
            inputs = features[:past_len]
            text_features = text_features[:past_len]


            inputs = torch.Tensor(inputs).to(device)
            text_features = torch.Tensor(text_features).to(device)

            inputs_embeds = inputs.unsqueeze(0)
            text_features = text_features.unsqueeze(0)
            outputs = model(inputs_embeds=inputs_embeds, text_inputs_embeds=text_features, labels_action=None, past_labels=None, labels_duration=None,return_preds=True)
            output_action = outputs['action']
            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # fine the forst none class
            none_mask = None
            for i in range(output_label.size(1)):
                if output_label[0, i] == NONE:
                    none_idx = i
                    break
                else:
                    none_idx = None

            if none_idx != 0:
                if none_idx is not None:
                    none_mask = torch.ones(output_label.shape).type(torch.bool)
                    none_mask[0, none_idx:] = False
                    none_mask = none_mask.to(output_action.device)

                output_dur = normalize_duration(output_dur, none_mask)

                pred_len = (0.5 + future_len * output_dur).squeeze(-1).long()

                pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
                predicted = torch.ones(future_len)
                action = output_label.squeeze()

                for i in range(len(action)):
                    predicted[int(pred_len[i]): int(pred_len[i] + pred_len[i + 1])] = action[i]
                    pred_len[i + 1] = pred_len[i] + pred_len[i + 1]
                    if i == len(action) - 1:
                        predicted[int(pred_len[i]):] = action[i]

                prediction = past_seq
                for i in range(len(predicted)):
                    prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[
                                                                  list(actions_dict_with_NONE.values()).index(
                                                                      predicted[i].item())]]))

            # evaluation
            for i in range(len(eval_p)):
                if none_idx != 0:
                    p = eval_p[i]
                    eval_len = int((obs_p + p) * vid_len)
                    eval_prediction = prediction[:eval_len]
                    T_action, F_action = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                    T_actions[i] += T_action
                    F_actions[i] += F_action
                else:
                    p = eval_p[i]
                    T_action, F_action = eval_file_all_none(gt_seq, obs_p, actions_dict, p)
                    T_actions[i] += T_action
                    F_actions[i] += F_action

        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i, j] != 0:
                    acc += float(T_actions[i, j] / total_actions[i, j])
                    n += 1

            result = 'obs. %d ' % int(100 * obs_p) + 'pred. %d ' % int(100 * eval_p[i]) + '--> MoC: %.4f' % (
                        float(acc) / n)
            description = "obs {} ,pred {}"
            key = description.format(obs_p, eval_p[i])
            moc = round(float(acc) / n, 4)
            moc_formatted = "{:.4f}".format(moc)
            res_des[key] = moc_formatted

            print(result)
        print('--------------------------------')

        return
