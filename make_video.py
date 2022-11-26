import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import sys
import matplotlib.pyplot as plt
from Datasets import ChairData
def make_video_from_images(images, write_to, fps = 24):
    """
    make a video from the rendered images
    :param images: the rendered images, torch tensor of shape (N, H, W, C)
    :param write_to: the output path
    :param fps: frames per second, default as 24
    :return:
    """
    # number of the images
    N = images.shape[0]
    seconds = N // fps
    total_frames = seconds * fps
    writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=fps)

    for i in range(total_frames):
        image = images[i].detach().cpu().numpy()
        # uint8_image = image.astype(np.uint8)
        writer.append_data(image)
    writer.close()

if __name__ == "__main__":
    root_dir = f'{sys.path[0]}/datasets/chair'
    write_to = f'./rendering_result.mp4'
    chair_data = ChairData(root_dir, mode='test')

    # Load the data
    data_loader = DataLoader(chair_data, batch_size=64, shuffle=False, num_workers=0)
    print(len(chair_data))
    images = []

    for i, (image, pose, focal) in enumerate(data_loader):
        images.append(image)

    # images = [image for (image, pose, focal) in data_loader]
    images = torch.cat(images, dim=0)
    make_video_from_images(images, write_to, fps=24)
    pass