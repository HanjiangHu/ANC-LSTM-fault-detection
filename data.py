import torch.utils.data as data
import os,glob,json,torch
import numpy as np



class FDI_train_dataset(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.input_path
        self.jsons = sorted(glob.glob(os.path.join(self.dataroot,'train')+'/*.json'))
        self.input_list = []
        self.label_list = []
        for json_file in self.jsons:
            with open(json_file, 'r') as f:
                pop_sequence_list = json.load(f)
                seq_input_list = [] # length is sequence
                seq_label_list = []
                for single_item in pop_sequence_list:
                    seq_input_list.append(np.array(single_item['input_list']))
                    seq_label_list.append(single_item['label'])
            f.close()
            self.input_list.append(torch.from_numpy(np.array(seq_input_list)))
            self.label_list.append(torch.from_numpy(np.array(seq_label_list)))


    def __getitem__(self, index):
        input_tensor = self.input_list[index]
        label = self.label_list[index]
        return {'input': input_tensor, 'label': label}

    def __len__(self):
        return len(self.jsons)

class FDI_val_dataset(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.input_path
        self.jsons = sorted(glob.glob(os.path.join(self.dataroot,'val')+'/*.json'))
        self.input_list = []
        self.label_list = []
        for json_file in self.jsons:
            with open(json_file, 'r') as f:
                pop_sequence_list = json.load(f)
                seq_input_list = [] # length is sequence
                seq_label_list = []
                for single_item in pop_sequence_list:
                    seq_input_list.append(np.array(single_item['input_list']))
                    seq_label_list.append(single_item['label'])
            f.close()
            self.input_list.append(torch.from_numpy(np.array(seq_input_list)))
            self.label_list.append(torch.from_numpy(np.array(seq_label_list)))


    def __getitem__(self, index):
        input_tensor = self.input_list[index]
        label = self.label_list[index]
        return {'input': input_tensor, 'label': label}

    def __len__(self):
        return len(self.jsons)

