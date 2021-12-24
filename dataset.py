""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import numpy as np

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset

class ReMiniImageNet:
    def __init__(self,args):
        self.original_path = args.original_miniimagenet_path
        self.modified_path = args.modified_miniimagenet_path
    
    def build(self):
        modified_train, modified_test = self.reshuffle_miniimagenet()
        self.save_to_modified_path(modified_train, modified_test)


    def save_to_modified_path(self, modified_train, modified_test):
        """Save to modified path"""
        with open(os.path.join(self.modified_path,'modified-mini-imagenet-train.pkl'), 'wb') as train_file:
            pickle.dump(modified_train, train_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.modified_path,'modified-mini-imagenet-test.pkl'), 'wb') as test_file:
            pickle.dump(modified_test, test_file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved both modified train, test to ',self.modified_path)

    def reshuffle_miniimagenet(self):
        self.id2numericallabel ={}
        with open(os.path.join(self.original_path, "mini-imagenet-classlabel.pkl"), "rb") as file:
            self.id2actuallabel = pickle.load(file)

        for idx,(k,_) in enumerate(self.id2actuallabel.items()):
            self.id2numericallabel[k] = idx 
        
        with open(os.path.join(self.original_path,'mini-imagenet-cache-train.pkl'), 'rb') as file:
            original_train_data = pickle.load(file)
        with open(os.path.join(self.original_path,'mini-imagenet-cache-val.pkl'), 'rb') as file:
            original_val_data = pickle.load(file)
        with open(os.path.join(self.original_path,'mini-imagenet-cache-test.pkl'), 'rb') as file:
            original_test_data = pickle.load(file)

        t_train_image, t_train_label, t_test_image, t_test_label = self.split_to_train_test(original_train_data)
        v_train_image, v_train_label, v_test_image, v_test_label = self.split_to_train_test(original_val_data)
        te_train_image, te_train_label, te_test_image, te_test_label = self.split_to_train_test(original_test_data)

        modified_train_image = np.concatenate([t_train_image,v_train_image,te_train_image])
        modified_train_label = t_train_label + v_train_label + te_train_label
        modified_test_image = np.concatenate([t_test_image,v_test_image,te_test_image])
        modified_test_label = t_test_label + v_test_label + te_test_label

        modified_train = {'image_data':modified_train_image, 'class_label':modified_train_label}
        modified_test = {'image_data':modified_test_image, 'class_label':modified_test_label}

        return modified_train, modified_test



    def split_to_train_test(self,data):
        """There are 600 images per class. Use the first 480(80%) as train, rest as test"""
        train_idx_list =[]
        test_idx_list =[]
        train_label =[]
        test_label =[]


        for k,v in data['class_dict'].items():
            train_idx_list.extend(v[:480])
            test_idx_list.extend(v[480:])
            train_label.extend([self.id2numericallabel[k]]*480)
            test_label.extend([self.id2numericallabel[k]]*120)
        assert len(train_idx_list) == len(train_label)
        assert len(test_idx_list) == len(test_label)
        
        train_image = data['image_data'][train_idx_list,:,:,:]
        test_image = data['image_data'][test_idx_list,:,:,:]

        return train_image, train_label, test_image, test_label
    




class MiniImageNet100(Dataset):
    def __init__(self, mode, data_path):
        self.modified_file = os.path.join(data_path,f'modified-mini-imagenet-{mode}.pkl')
        with open(self.modified_file, 'rb') as file:
            self.data = pickle.load(file)

        # define dataset length
        self.length = self.data['image_data'].shape[0]
        # define transform module
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

    def __len__(self):
        """Return number of total samples
        Returns: number of total samples
        """
        return self.length

    def __getitem__(self, index):
        image = self.data['image_data'][index,:,:,:]
        label = self.data['class_label'][index]

        if self.transform:
            image = self.transform(image)
        return image,label
