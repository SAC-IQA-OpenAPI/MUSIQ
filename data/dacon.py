import os
import torch
import numpy as np
import cv2
import pandas as pd


class DaconDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, csv_file_name, scale_1, scale_2, transform, train_mode, scene_list, train_size=0.8):
        super(DaconDataset, self).__init__()
        
        self.db_path = db_path
        self.csv_file_name = csv_file_name
        self.scale_1 = scale_1
        self.scale_2 = scale_2
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size

        self.data_dict = CSVDatalist(
            csv_file_path = self.csv_file_name,
            train_mode = self.train_mode,
            scene_list = self.scene_list,
            train_size = self.train_size
        ).load_data_dict()
        
        self.n_images = len(self.data_dict['d_img_list'])
        
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        # d_img_org : H x W x C
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_org = cv2.imread(os.path.join(self.db_path, d_img_name) + '.jpg', cv2.IMREAD_COLOR)
        # print(os.path.join(self.db_path, d_img_name) + '.jpg', "os.path.join(self.db_path, d_img_name) + '.jpg'")
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        d_img_org = np.array(d_img_org).astype('float32') / 255.0
        
        d_img_org = cv2.resize(d_img_org, dsize=(1024, 768), interpolation=cv2.INTER_CUBIC)
        h, w, c = d_img_org.shape
        # print(d_img_org.shape)
        d_img_scale_1 = cv2.resize(d_img_org, dsize=(self.scale_1, int(h * self.scale_1 / w)), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = cv2.resize(d_img_org, dsize=(self.scale_2, int(h * self.scale_2 / w)), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = d_img_scale_2[:160, :, :] # height = 160까지의 image
        
        score = self.data_dict['score_list'][idx]
        sample = {'d_img_org' : d_img_org, 'd_img_scale_1' : d_img_scale_1, 'd_img_scale_2' : d_img_scale_2, 'score' : score}
        
        if self.transform:
            sample = self.transform(sample)
        # for key, value in sample.items():
        #     if key == 'd_img_org' or key == 'd_img_scale_1' or key == 'd_img_scale_2':
        #         print(key, value.shape)
        return sample
        

class CSVDatalist():
    def __init__(self, csv_file_path, train_mode, scene_list, train_size=0.8):
        self.csv_file_path = csv_file_path
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list
    
    def load_data_dict(self):
        scn_idx_list, d_img_list, score_list = [], [], []
                
        csv_file = pd.read_csv(self.csv_file_path)
        img_name_list = csv_file['img_name']
        mos_list = csv_file['mos']
        mos_list = np.array(mos_list)
        mos_list = mos_list.astype('float').reshape(-1, 1)
        data_dict = {'d_img_list' : img_name_list, 'score_list' : mos_list}
        return data_dict
                