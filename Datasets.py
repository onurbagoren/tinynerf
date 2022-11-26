from __future__ import division, print_function

import os
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms, utils
import json
from PIL import Image
import imageio
import sys
from typing import Any
import load_liff

class TRex(Dataset):
    '''
    Dataloader for the trext dataset
    '''

    def __init__(self, root_dir) -> None:
        super().__init__()
        images, poses, _, render_poses, i_test = load_liff.load_llff_data(root_dir)
        self.focal_length = 3225.60 # from the dataset

        homog_stuff = torch.zeros((55, 1, 4))
        homog_stuff[:, :, 3] = 1 # homogeneous coordinate
        poses = poses[:,:,:4]
        # Add the homogenous coordinates to each of the poses
        final_poses = torch.cat((poses, homog_stuff), dim=1)
        images_first = images[:i_test, :, :, :]
        images_second = images[i_test+1:, :, :, :]
        self.images = torch.cat((images_first, images_second), dim=0)
        
        poses_first = poses[:i_test, :, :]
        poses_second = poses[i_test+1:, :, :]
        self.final_poses = torch.cat((poses_first, poses_second), dim=0)

        self.i_test = i_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
        ])
        return transform(self.images[index].permute(2, 0, 1)), self.final_poses[index], self.focal_length

class ChairData(Dataset):
    '''
    Dataloader for the chairs dataset
    '''

    def __init__(self, root_dir, mode='train') -> None:
        super().__init__()

        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, f'transforms_{mode}.json'), 'r') as f:
            self.transform = json.load(f)
        self.mode = mode
        self.images = os.listdir(os.path.join(self.root_dir, mode))
        if mode == 'test':
            images = []
            for im_name in self.images:
                if 'depth' in im_name:
                    continue
                images.append(im_name)
            self.images = images
        self.camera_angle_x = self.transform['camera_angle_x']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform_information = self.transform['frames'][idx]
        file_path = transform_information['file_path'][2:] + '.png'
        rotation = transform_information['rotation']
        transformation_matrix = torch.from_numpy(
            np.array((transform_information['transform_matrix'])))
        image_dir = os.path.join(self.root_dir)
        image = Image.open(os.path.join(image_dir, file_path))
        image = image.convert('RGB')
        H, W = image.size
        # Resize image to 100 x 100
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        image = transform(image).permute(1, 2, 0)
        H, W, C = image.shape
        focal = .5 * W / np.tan(.5 * self.camera_angle_x)
        return image, transformation_matrix, focal


def main():
    root_dir = f'{sys.path[0]}/datasets/chair'
    images_dir = f'{root_dir}/images'

    trex_data = ChairData(root_dir, mode='test')

    # Load the data
    data_loader = DataLoader(trex_data, batch_size=1, shuffle=False, num_workers=0)
    for i, (image, pose, focal) in enumerate(data_loader):
        print(image.shape)
        plt.imshow(image[0, ...])
        plt.show()
        break


if __name__ == '__main__':
    main()
