import random
import torch.utils.data as Data
import os
import numpy as np
import torch
import torch.nn.functional as F



class ActionDataset(Data.Dataset):
    def __init__(self, args, mode,vid_list,actions_dict,features_path,text_feature_path,gt_path,dataset,n_class, pad_idx):
        super(ActionDataset, self).__init__()
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------

        self.args = args
        self.mode = mode
        self.dataset=dataset

        self.vid_list = list()
        self.actions_dict = actions_dict
        self.features_path = features_path
        self.text_feature_path = text_feature_path
        self.gt_path = gt_path
        self.sample_rate = args.sample_rate
        self.pred_ratio = args.pred_ratio

        # fix query
        self.n_class = n_class
        self.pad_idx = pad_idx
        self.NONE = self.n_class - 1
        self.n_query = args.n_query

        print(f"number of {self.mode} videos: {len(vid_list)}\n")

        if self.mode=='train':
            if self.dataset == 'breakfast':
                start_frame = 0
                for vid in vid_list:
                    self.vid_list.append([vid, .2, start_frame])
                    self.vid_list.append([vid, .3, start_frame])
                    self.vid_list.append([vid, .5, start_frame])
            elif self.dataset == '50_salads':
                for vid in vid_list:
                    for i in range(1, 6):    # [0.1, 0.2, 0.3, 0.4, 0.5]  max: 0.5
                        value = i / 10
                        for start_frame in range(args.sample_rate):   # strar fram :
                            self.vid_list.append([vid, value, start_frame])
        elif self.mode=='val':
            for vid in vid_list:
                start_frame = 0
                self.vid_list.append([vid, .2, start_frame])
                self.vid_list.append([vid, .3, start_frame])

        self._make_input(vid, 0.2, 0)

    def _make_input(self, vid_file, obs_perc, start_frame):
        vid_file = vid_file.split('/')[-1]    #video name

        gt_file = os.path.join(self.gt_path, vid_file)
        feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+'.npy')
        features = np.load(feature_file)   #[2048,12040]
        features = features.transpose()    #[12040,2048]
        features = features[::self.sample_rate]   # extract frame

        text_feature_file = os.path.join(self.text_feature_path, vid_file.split('.')[0] + '.npy')
        text_features = np.load(text_feature_file)  # [2048,12040]

        file_ptr = open(gt_file, 'r')
        all_content = file_ptr.read().split('\n')[:-1]
        all_content = all_content[::self.sample_rate]    # extract frame

        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)   #fix query need to do

        # past feature
        features = features[start_frame : start_frame + observed_len] #[S, C]
        text_features = text_features[start_frame: start_frame + observed_len]

        #past label
        past_content = all_content[start_frame: start_frame + observed_len]  # [S]
        past_label = self.seq2idx(past_content)
        past_label = torch.Tensor(past_label).long()

        if np.shape(features)[0] != len(past_label) :
            features = features[:len(past_label),]

        if np.shape(text_features)[0] != len(past_label) :
            text_features = text_features[:len(past_label),]


        #future label
        future_content = all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)  # 将 self.NONE 添加到 trans_future 数组的末尾
        trans_future_target = trans_future  # target


        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0:
            tmp = np.ones(diff) * self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff + 1) * self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0:
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else:
            tmp_len = np.ones(1) * self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))

        trans_future_target = torch.Tensor(trans_future_target).long()


        item = {'inputs_embeds':torch.Tensor(features),
                'text_inputs_embeds': torch.Tensor(text_features),
                'labels_action': torch.Tensor(trans_future_target),
                'past_labels': torch.Tensor(past_label),
                'labels_duration': torch.Tensor(trans_future_dur)
                }

        return item

    def __getitem__(self, idx):
        vid_file, obs_perc, start_frame = self.vid_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc, start_frame)

        return item

    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['inputs_embeds'] for item in batch]
        b_text_features = [item['text_inputs_embeds'] for item in batch]
        b_labels_action = [item['labels_action'] for item in batch]
        b_past_labels=[item['past_labels'] for item in batch]
        # b_attention_mask = [item['attention_mask'] for item in batch]
        b_labels_duration = [item['labels_duration'] for item in batch]

        sizes = [t.shape[0] for t in b_past_labels]
        max_size = max(sizes)

        sequence_lengths = [len(seq) for seq in b_features]    # original length
        padding_lengths = [max_size - length for length in sequence_lengths]   # length needed to fill

        # fill in front
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (0,0,padding,0), value=0) for seq, padding in zip(b_features, padding_lengths)], batch_first=True,
            padding_value=0)

        text_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (0, 0, padding, 0), value=0) for seq, padding in zip(b_text_features, padding_lengths)],batch_first=True,
            padding_value=0)

        past_labels = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (padding, 0), value=-100) for seq, padding in zip(b_past_labels, padding_lengths)], batch_first=True,
            padding_value=-100)

        labels_action = torch.nn.utils.rnn.pad_sequence(b_labels_action, batch_first=True, padding_value=self.pad_idx)
        labels_duration = torch.nn.utils.rnn.pad_sequence(b_labels_duration, batch_first=True,padding_value=self.pad_idx)

        batch = [inputs_embeds, text_inputs_embeds, past_labels, labels_action, labels_duration]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def shuffle_list(self, list):
        random.shuffle(list)

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)

