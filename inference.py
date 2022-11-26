import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Datasets
from nerf_utils import *
from make_video import *
import numpy as np
import imageio
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = f'{sys.path[0]}/datasets/chair'

chairs_test = Datasets.ChairData(root_dir, mode='test')
test_dataloader = torch.utils.data.DataLoader(
    chairs_test, batch_size=1, shuffle=False)

# # make a video for the groundtruth data
# write_to = f'./rendering_groundtruth_result.mp4'
# images = []
# for i, (image, pose, focal) in enumerate(test_dataloader):
#     images.append(image)
#
# # images = [image for (image, pose, focal) in data_loader]
# images = torch.cat(images, dim=0)
# make_video_from_images(images, write_to, fps=24)

model = torch.jit.load('./results/model_scripted.pt')

num_test_images = 200
predicted_images = []
for ii in range(num_test_images):
    print(f"number {ii} images")
    train_img, transform, focal = chairs_test[ii]
    print(train_img.shape)
    # print(transform)
    # Move data to device
    H, W, _ = train_img.shape
    train_img = train_img.to(device)
    transform = transform.to(device)
    focal = torch.tensor(focal, device=device)

    # rand_int = np.random.randint(train_images.shape[0])
    # train_img = train_images[rand_int]
    # transform = train_transforms[rand_int]
    # focal = focal_lengths

    # Generate inputs for the NeRF model
    network_input = generate_network_input(train_img, focal, transform)
    encoded_points, query_points, depth_values = network_input
    # Get the output of the model
    # H, W, num_samples, D = encoded_points.shape
    # batch_of_rays = torch.chunk(encoded_points.reshape(-1, encoded_points.shape[-1]), 4096, dim=0)
    # output = []
    # for batch in batch_of_rays:
    #     batch_out = model(batch)
    #     output.append(batch_out)
    #
    # output = torch.cat(output, dim=0)
    # output = output.reshape((H, W, num_samples, -1))


    output = model(encoded_points)
    # Predict the color on the image frame
    predicted_rgb = render_volume_density(
        output, query_points, depth_values)
    predicted_rgb = predicted_rgb.detach().cpu().numpy()
    predicted_images.append(predicted_rgb)
print("Done")
fps=24
predicted_images = np.array(predicted_images)
N = predicted_images.shape[0]
seconds = N // fps
total_frames = seconds * fps
write_to = f'./rendering_predicted_result.mp4'
writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=fps)

for i in range(total_frames):
    image = predicted_images[i]
    # uint8_image = image.astype(np.uint8)
    writer.append_data(image)
writer.close()
print("Done")