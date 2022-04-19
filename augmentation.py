import os
import numpy as np
import pandas as pd
import re

import tsaug
from tsaug.visualization import plot

'''
Segmentation of time series and addition of noise
'''

old_path = './data_fMRI'
save_path = 'data_augm'

for root_dir, sub_dirs, _ in os.walk(old_path):
    for sub_dir in sub_dirs:
        save_dir = os.path.join(save_path, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            file_names = os.listdir(save_dir)
            for files in file_names:
                c_path = os.path.join(save_dir, files)
                os.remove(c_path)
        file_names = os.listdir(os.path.join(root_dir, sub_dir))
        file_names = list(filter(lambda x: x != 'ROI_CenterOfMass.xlsx', file_names))
        for file in file_names:

             
            read_path = os.path.join(old_path, sub_dir, file)
            init_data = pd.read_excel(read_path, header=None)
            n = np.array(init_data)
            noised_init_data = tsaug.AddNoise(scale=0.12).augment(n.T)
            noised_init_data = noised_init_data.T
            noised_init_data2 = tsaug.Convolve(window="hann", size=5).augment(n.T)
            noised_init_data2 = noised_init_data2.T
            # X = np.array(init_data)
            # plot(X)
            n = int(init_data.shape[0] / 5)
            for i in range(5):
                single_data = init_data[i * n:(i + 1) * n]
                noised_single_data = noised_init_data[i * n:(i + 1) * n]
                noised_single_data2 = noised_init_data2[i * n:(i + 1) * n]
                file_save_name = re.sub(r'.xlsx', '_%i.npy' % i, file)
                noised_file_save_name = re.sub(r'.xlsx', '_noised_%i.npy' % i, file)
                noised_file_save_name2 = re.sub(r'.xlsx', '_noised2_%i.npy' % i, file)
                 
                np.save(os.path.join(save_dir, file_save_name), single_data)
                np.save(os.path.join(save_dir, noised_file_save_name), noised_single_data)
                np.save(os.path.join(save_dir, noised_file_save_name2), noised_single_data2)
