from ast import Num
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


class Nerf(nn.Module):
    '''
    NERF (Neural Render Field) architecture
    '''

    def __init__(self, input_dim):
        '''
        Initialize the architecture. Will be a smaller version of the paper.
        The arcitecture follows (nn.Linear -> nn.ReLU) -- it's just and MLP!

        Parameters
        ----------
        input_dim : int
            The dimension of the input. Should be 2 * 3 * encoding_functions
                (2 for sin, cos of the encoding and 3 for x, y, z)
        '''
        super(Nerf, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        '''
        Forward pass through the model.
        '''
        return self.model(x.float())

def train_synthetic(train_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    model: Nerf,
                    optimizer: torch.optim,
                    num_iterations: int,
                    device: str,
                    log_interval_iterations: int,
                    plot_intervals: int,
                    fixed_test: bool = False) -> Nerf:
    '''
    Function for training the model.    

    Parameters
    ----------
    train_dataloader : DataLoader
        The dataloader for the training set
    test_dataloader : DataLoader
        The dataloader for the test set
    model : Nerf
        The model to train
    optimizer : torch.optim
        The optimizer to use
    num_iterations : int
        The number of num_iterations to train for
    device : str
        The device to use
    log_interval : int
        The interval to log the training data
    scheduler : torch.optim.lr_scheduler
        The scheduler to use
    fixed_test : bool
        Whether to use the fixed test set

    Returns
    -------
    Nerf
        The trained model
    '''
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H_%M_%S")
    writer = SummaryWriter('runs/' + now_str)

    # Generate fixed test image
    if fixed_test:
        test_img, test_pose, test_focal = next(iter(test_dataloader))
        test_img = test_img[0, ...].to(device)
        test_pose = test_pose[0, ...].to(device)
        test_focal = test_focal.to(device)
    # Start training the model

    train_losses = []
    test_losses = []
    ssims = []
    psnrs = []

    for ii in range(num_iterations):
        train_img, transform, focal = next(iter(train_dataloader))

        # Move data to device
        _, H, W, _ = train_img.shape
        train_img = train_img[0, ...].to(device)
        transform = transform[0, ...].to(device)
        focal = focal[0].to(device)

        # Generate inputs for the NeRF model
        network_input = generate_network_input(train_img, focal, transform)
        encoded_points, query_points, depth_values = network_input
        
        output = model(encoded_points)

        # Predict the color on the image frame
        predicted_rgb = render_volume_density(
            output, query_points, depth_values)

        # Compute the loss
        train_loss = F.mse_loss(predicted_rgb, train_img)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar('loss/train', train_loss, ii)

        if ii % log_interval_iterations == 0:
            # Log training loss
            print('Iteration: {}\tLoss: {}'.format(ii, train_loss))
            train_losses.append(train_loss.item())

            # Generate inputs for the NeRF model
            network_input = generate_network_input(test_img, focal,
                                                   test_pose)
            encoded_points, query_points, depth_values = network_input
            # Get the output of the model
            output = model(encoded_points)

            # Predict the color on the image frame
            predicted_rgb = render_volume_density(
                output, query_points, depth_values)

            test_loss = F.mse_loss(predicted_rgb, test_img)
            test_losses.append(test_loss.item())
            writer.add_scalar('loss/test', test_loss, ii)

            pred_gray = predicted_rgb.detach().cpu().numpy()
            test_gray = test_img.detach().cpu().numpy()

            ssim_val = ssim(pred_gray, test_gray, data_range=1, channel_axis=2)
            writer.add_scalar('ssim/test', ssim_val, ii)
            ssims.append(ssim_val)

            # Compute PSNR
            monochrome_orig = torch.mean(test_img, dim=-1).reshape(-1, 1)
            max_monochrome, _ = torch.max(monochrome_orig, dim=0)

            psnr = 20 * torch.log10(max_monochrome) - 10 * torch.log10(
                test_loss)
            writer.add_scalar('psnr/test', psnr, ii)
            psnrs.append(psnr.item())

        if ii % plot_intervals == 0:
            # Generate inputs for the NeRF model
            network_input = generate_network_input(test_img, focal,
                                                   test_pose)
            encoded_points, query_points, depth_values = network_input
            # Get the output of the model
            output = model(encoded_points)

            # Predict the color on the image frame
            predicted_rgb = render_volume_density(
                output, query_points, depth_values)
                
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].imshow(test_img.cpu().detach().numpy())
            axs[1].imshow(predicted_rgb.cpu().detach().numpy())
            axs[0].set_title('Original')
            axs[1].set_title('Predicted')
            plt.savefig(f'results/{ii}.png'.format(now_str, ii))
            plt.close(fig)

    return model, train_losses, test_losses, ssims, psnrs


def generate_network_input(
        img: torch.Tensor,
        focal_length: torch.float64,
        transform: torch.Tensor) -> torch.Tensor:
    '''
    Generate the network input for the model.

    Args:
        img: (H, W, 3) torch Tensor
            Image to use for the rendering. Only used to extract the height and
            width.
        focal_length: torch.float64
            The focal length of the camera, from the dataset
        transform: (4, 4) torch Tensor
            The transform matrix to use for the rendering, describing the pose
            of the camera in the world frame.

    Returns:
        encoded_points: (N, 3) torch Tensor
    '''
    # Get the image shape
    H, W, _ = img.shape

    # Get the query points
    query_points, depth_values = ray_and_query_points(
        H, W, focal_length, transform, near_thresh=2, far_thresh=6, num_samples=32)

    # Get the positional encoding
    encoded_points = positional_encoding(query_points)

    # Return the network input
    return encoded_points, query_points, depth_values


def ray_and_query_points(height: int, width: int, focal_length: float, cam2world: torch.Tensor,
                         near_thresh: float, far_thresh: float, num_samples: int):

    # creating a coordinate system to index into each pixel
    y, x = torch.meshgrid(
        torch.arange(width).to(cam2world),
        torch.arange(height).to(cam2world))

    # Z is negativ according to convention
    z = (-1)*torch.ones_like(x)

    # shifting the coordinate system of x and y to the center
    x_centered = x - (width/2)
    y_centered = (height/2) - y

    # scaling the size of the coordinates
    x_scaled = x_centered/focal_length
    y_scaled = y_centered/focal_length

    # Combining x,y and z to a height*width*3 matrix.
    # Each pixel center in the image has 3 coordinates
    directions = torch.stack([x_scaled, y_scaled, z], dim=-1)

    # origin from cam2world matrix
    origin = cam2world[:3, -1]

    # Same origin for each pixel
    ray_origins_output = origin.repeat([height, width, 1])
    ray_origins = origin.repeat([height, width, num_samples, 1])
    # reshaping for calcuation purposes
    ray_origins = torch.reshape(
        ray_origins, (height, width, num_samples, 1, 3))

    # query points
    height = ray_origins.shape[0]
    width = ray_origins.shape[1]

    # Using ray_directions as the beginning of the ray excluding the origin
    ray_beginning = directions
    # reshaping for calcuation purposes
    ray_beginning = torch.reshape(ray_beginning, (height, width, 1, 3))

    # by repeating the image num_samples times we have created a ray along the same direction
    # as the image pixel. We have each pixel num_samples times. We later scale each ray
    # by the depth values
    ray_beginning = ray_beginning.unsqueeze(2).repeat(1, 1, num_samples, 1, 1)

    # create the step interval
    interval = (far_thresh - near_thresh)/num_samples

    # create a vector of all z values from the beginning to the end of the ray per interval
    # if the interval is 32 then we split it evenly into 32 parts between near_thres and far_thresh
    ray_interval = torch.arange(
        near_thresh, far_thresh, interval).to(cam2world)
    ray_points_z_out = ray_interval.repeat([height, width, 1])

    # reshaping for calcuation purposes
    ray_points_z = torch.reshape(
        ray_interval, (num_samples, 1, 1)).to(cam2world)

    # we are now scaling each pixel with their corresponding query point to get the query points along
    # each ray
    query_points_camera = ray_beginning*ray_points_z
    # since we are still in the camera frame, we have to shift all points into the world frame
    query_points_world = torch.matmul(query_points_camera, cam2world[:3, :3].T)
    # we then add the origin to each ray
    query_points = query_points_world + ray_origins
    query_points = torch.reshape(query_points, (height, width, num_samples, 3))

    return query_points, ray_points_z_out


def positional_encoding(
        query_points: torch.Tensor,
        num_encoding_functions: int = 6,
        include_input: bool = True
) -> torch.Tensor:
    """
    encode the input tensor, the input tensor should be an (N, 3), and the output
    tensor should be (N, 2 * 3 * num_encoding_functions + 3 (if include_input is True))

    Args:
        query_points: (H, W, num_samples, 3) torch Tensor
            Points to sample across the rays
        num_encoding_functions: int
            The number of functions to use to encode the x,y,z positions
        include_input:  bool
            Whether to include the input in the encoding

    Returns:
        encoded_tensor: (H, W, num_samples, M) torch Tensor
            The encoded tensor, M = (2 * 3 * num_encoding_functions) or
            (2 * 3 * num_encoding_functions + 3) if include_input is True
    """
    H, W, num_samples, D = query_points.shape
    N = H * W * num_samples
    flattened_query_points = query_points.view(N, D)

    # generate L encode functions
    func_repeat = torch.linspace(
        0.0, num_encoding_functions - 1, num_encoding_functions)
    frequency_bands = 2.0 ** func_repeat.to(flattened_query_points)

    # repeat query points N times as the input of sin and cos
    # (N, 1, [x, y, z, x, y, z])
    tensor_repeat = flattened_query_points.repeat(
        (1, 2)).reshape((N, -1, 2 * D))
    # repeat encode functions N times
    frequency_repeat = frequency_bands.repeat(
        (N, 1)).reshape((N, num_encoding_functions, 1))
    encoded_tensor = torch.bmm(frequency_repeat, tensor_repeat)

    # apply sin and cos to the encoded tensors
    encoded_tensor[..., 0:3] = torch.sin(encoded_tensor[..., 0:3])
    encoded_tensor[..., 3:] = torch.cos(encoded_tensor[..., 3:])
    encoded_tensor = encoded_tensor.reshape((N, -1))

    # include the raw inputs
    if include_input:
        encoded_tensor = torch.cat(
            [flattened_query_points, encoded_tensor], -1)

    # Reshape the encoded tensor to (H, W, num_samples, 2 * 3 * num_encoding_functions + 3 (if include_input is True))
    encoded_tensor = encoded_tensor.reshape((H, W, num_samples, -1))

    return encoded_tensor


def render_volume_density(model_output, query_points, depth_values):
    '''
    Volume rendering for the NeRF model.

    Args:
        model_output: (H, W, num_samples, M) torch Tensor
            The output of the model, M = (2 * 3 * num_encoding_functions) or
            (2 * 3 * num_encoding_functions + 3) if include_input is True
        query_points: (H, W, num_samples, 3) torch Tensor
            Points to sample across the rays
        depth_values: (H, W, num_samples) torch Tensor
            The depth values for each ray

    Returns:
        rendered_volume: (H, W, 3) torch Tensor
            The RGB image output of the volume rendering
    '''

    volume_function = model_output[..., 3]
    colors = model_output[..., :3]
    H, W, num_samples, _ = query_points.shape
    C = torch.zeros((H, W, 3)).to(model_output)

    for p in range(2, num_samples):

        # Extracting the current and previous depth value
        curr_point = depth_values[:, :, p]
        prev_points = depth_values[:, :, p-1]

        # Getting their difference
        delta = (curr_point - prev_points).to(model_output)
        # Clipping above 1
        sigma = torch.nn.functional.relu(
            volume_function[:, :, p-1]).to(model_output)
        # Calculating alpha
        val = -1 * delta * sigma
        alpha = 1. - torch.exp(val)

        # Compute color
        color = colors[:, :, p, :]
        # Computing the accumulated transmittance T
        prev_trans_pos = depth_values[:, :, 0:p-1]
        trans_pos = depth_values[:, :, 1:p]
        trans_delta = (trans_pos - prev_trans_pos).to(model_output)
        sigma = volume_function[:, :, :p-1]
        sigma = torch.nn.functional.relu(sigma).to(model_output)
        sum = torch.sum(sigma * trans_delta, dim=2)
        T = torch.exp(-sum)

        # Calculating the color for each ray in each pixel
        # T is of size (height, width), alpha (height, width) and color (height, width, 3(rgb))
        intermediate = T*alpha
        # reshaping for calculation purposes
        intermediate = intermediate.unsqueeze(2).repeat(1, 1, 3)
        # C will be a (height, width, rgb) matrix
        C += intermediate * color
    return C
