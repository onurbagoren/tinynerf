import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import nerf_utils
from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Initially just attempting toe visualize the plane that is the image plane

def visualize_ray_casting(height, width, near_threshold, far_threshold, num_samples, c2w, fig):
    '''
    Function to visualize the transformation of the camera from it's own frame to the 
    world frame.
    Useful for visualizing the ray casting process.

    Args:
        height: height of the image plane
        width: width of the image plane
        near_threshold: distance from the camera to the near plane
        far_threshold: distance from the camera to the far plane
        num_samples: number of samples to take along the ray
        c2w: camera to world transformation matrix
    '''

    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    x = torch.linspace(0, width, width)
    y = torch.linspace(0, height, height)

    YY, XX = torch.meshgrid(x, y)
    XX = XX - width/2
    YY = height/2 - YY
    ZZ = -1 * torch.ones_like(XX)

    focal_length = 1
    num_samples = 4

    X_scaled = XX / focal_length
    Y_scaled = YY / focal_length

    directions = torch.stack([X_scaled, Y_scaled, ZZ], dim=2)

    origin = torch.tensor([0, 0, 0])

    ray_origins_output = origin.repeat([height, width, 1])

    ray_beginning = directions.view([height, width, 1, 3])
    ray_beginning = ray_beginning.unsqueeze(
        2).repeat((1, 1, num_samples, 1, 1))

    interval = (far_threshold - near_threshold)/num_samples

    # create a vector of all z values from the beginning to the end of the ray per interval
    # if the interval is 32 then we split it evenly into 32 parts between near_thres and far_thresh
    ray_points_z_output = torch.arange(
        near_threshold, far_threshold, interval)

    # reshaping for calcuation purposes
    ray_points_z = torch.reshape(
        ray_points_z_output, (num_samples, 1, 1))

    # we are now scaling each pixel with their corresponding query point to get the query points along
    # each ray
    query_points_camera = ray_beginning*ray_points_z
    query_points_camera_viz = query_points_camera.view(
        [height, width, num_samples, 3])
    query_points_beginning = query_points_camera_viz[:, :, 0, :]
    query_points_end = query_points_camera_viz[:, :, -1, :]
    # Draw a line between the first and last query points

    flattened_origin = ray_origins_output.view([height * width, 3])
    flattened_x = flattened_origin[:, 0]
    flattened_y = flattened_origin[:, 1]
    flattened_z = flattened_origin[:, 2]

    flattened_beginning_point = query_points_beginning.view(
        [height * width, 3])
    flattened_beginning_x = flattened_beginning_point[:, 0]
    flattened_beginning_y = flattened_beginning_point[:, 1]
    flattened_beginning_z = flattened_beginning_point[:, 2]

    flattened_end_point = query_points_end.view([height * width, 3])
    flattened_end_x = flattened_end_point[:, 0]
    flattened_end_y = flattened_end_point[:, 1]
    flattened_end_z = flattened_end_point[:, 2]

    ax.scatter(ray_origins_output[:, :, 0], ray_origins_output[:, :, 1],
               ray_origins_output[:, :, 2], color='r', label='Ray origin')
    ax.scatter(XX, YY, -1 * ZZ, color='b', label='Image Plane')
    for ii in range(height*width):
        if ii == 0:
            ax.plot([flattened_beginning_x[ii], flattened_end_x[ii]], [flattened_beginning_y[ii], flattened_end_y[ii]], [
                    flattened_beginning_z[ii], flattened_end_z[ii]], color='b', alpha=0.2, label='Query vector')
            ax.plot([flattened_beginning_x[ii], flattened_x[ii]], [flattened_beginning_y[ii], flattened_y[ii]], [
                    flattened_beginning_z[ii], flattened_z[ii]], color='y', alpha=0.2, label='Origin to query vector')
            ax.plot([flattened_x[ii], XX.flatten()[ii]], [flattened_y[ii], YY.flatten()[ii]], [
                    flattened_z[ii], -1 * ZZ.flatten()[ii]], color='r', alpha=0.2, label='Image plane to origin')
        ax.plot([flattened_beginning_x[ii], flattened_end_x[ii]], [flattened_beginning_y[ii], flattened_end_y[ii]], [
                flattened_beginning_z[ii], flattened_end_z[ii]], color='b', alpha=0.2)
        ax.plot([flattened_beginning_x[ii], flattened_x[ii]], [flattened_beginning_y[ii], flattened_y[ii]], [
                flattened_beginning_z[ii], flattened_z[ii]], color='y', alpha=0.2)
        ax.plot([flattened_x[ii], XX.flatten()[ii]], [flattened_y[ii], YY.flatten()[
                ii]], [flattened_z[ii], -1 * ZZ.flatten()[ii]], color='r', alpha=0.2)

    for n in range(num_samples):
        if n == 0:
            ax.scatter(query_points_camera_viz[:, :, n, 0], query_points_camera_viz[:, :, n, 1],
                       query_points_camera_viz[:, :, n, 2], color='g', alpha=0.2, label='Sample points')
        ax.scatter(query_points_camera_viz[:, :, n, 0], query_points_camera_viz[:,
                   :, n, 1], query_points_camera_viz[:, :, n, 2], color='g', alpha=0.2)

    # Prep values for rotation and whatnor
    prep_dir = directions.permute(2, 0, 1).flatten(start_dim=1)
    ones = torch.ones((1, height * width))
    prep_dir = torch.concat((prep_dir, ones), dim=0)

    ones = torch.ones((height, width, 4, 1, 1))
    new_query_points_world = torch.concat((query_points_camera, ones), dim=-1)
    query_points_world = torch.matmul(new_query_points_world, c2w.T)
    new_directions = torch.stack(
        [X_scaled, Y_scaled, -1 * ZZ, torch.ones_like(X_scaled)], dim=-1)
    rotated_directions = torch.matmul(new_directions, c2w.T)
    query_points_world = query_points_world.view(
        height, width, num_samples, 4)[:, :, :, :3]
    world_query_points_beginning = query_points_world[:, :, 0, :]
    world_query_points_end = query_points_world[:, :, -1, :]
    new_origin = c2w[:3, 3]
    world_ray_origins_output = new_origin.repeat([height, width, 1])

    ax2.scatter(world_ray_origins_output[:, :, 0], world_ray_origins_output[:,
                :, 1], world_ray_origins_output[:, :, 2], color='r', label='Ray origin')
    ax2.scatter(rotated_directions[:, :, 0], rotated_directions[:, :, 1],
                rotated_directions[:, :, 2], color='b', label='Image Plane')

    beginning_flat = world_query_points_beginning.view([height * width, 3])
    end_flat = world_query_points_end.view([height * width, 3])
    beg_w_x = beginning_flat[:, 0]
    beg_w_y = beginning_flat[:, 1]
    beg_w_z = beginning_flat[:, 2]
    end_w_x = end_flat[:, 0]
    end_w_y = end_flat[:, 1]
    end_w_z = end_flat[:, 2]
    flatten_dirs = rotated_directions.view([height * width, 4])
    flattened_ray_origins_output = world_ray_origins_output.view(
        [height * width, 3])

    for ii in range(height * width):
        if ii == 0:
            ax2.plot([beg_w_x[ii], end_w_x[ii]], [beg_w_y[ii], end_w_y[ii]], [
                     beg_w_z[ii], end_w_z[ii]], color='b', alpha=0.2, label='Query vector')
            ax2.plot([flattened_ray_origins_output[ii, 0], beg_w_x[ii]], [flattened_ray_origins_output[ii, 1], beg_w_y[ii]], [
                     flattened_ray_origins_output[ii, 2], beg_w_z[ii]], color='y', alpha=0.6, label='Origin to query vector')
            ax2.plot([flattened_ray_origins_output[ii, 0], flatten_dirs[ii, 0]], [flattened_ray_origins_output[ii, 1], flatten_dirs[ii, 1]], [
                     flattened_ray_origins_output[ii, 2], flatten_dirs[ii, 2]], color='r', alpha=0.2, label='Image plane to origin')
        ax2.plot([beg_w_x[ii], end_w_x[ii]], [beg_w_y[ii], end_w_y[ii]], [
                 beg_w_z[ii], end_w_z[ii]], color='b', alpha=0.2)
        ax2.plot([flattened_ray_origins_output[ii, 0], beg_w_x[ii]], [flattened_ray_origins_output[ii, 1],
                 beg_w_y[ii]], [flattened_ray_origins_output[ii, 2], beg_w_z[ii]], color='y', alpha=0.6)
        ax2.plot([flattened_ray_origins_output[ii, 0], flatten_dirs[ii, 0]], [flattened_ray_origins_output[ii, 1],
                 flatten_dirs[ii, 1]], [flattened_ray_origins_output[ii, 2], flatten_dirs[ii, 2]], color='r', alpha=0.2)

    for n in range(num_samples):
        if n == 0:
            ax2.scatter(query_points_world[:, :, n, 0], query_points_world[:, :, n, 1],
                        query_points_world[:, :, n, 2], color='g', alpha=0.4, label='Sample points')
        ax2.scatter(query_points_world[:, :, n, 0], query_points_world[:,
                    :, n, 1], query_points_world[:, :, n, 2], color='g', alpha=0.4)

    # Axis titles
    ax.set_title('Query points in camera frame')
    ax.legend()
    ax2.legend()
    ax2.set_title('Query points in world frame')


def main():
    H = 10
    W = 10
    num_samples = 6
    near_threshold = 2
    far_threshold = 10
    rot = R.from_euler('xyz', [45, 45, 45], degrees=True)
    t = torch.tensor([4, -2, 10])
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(rot.as_matrix())
    c2w[:3, 3] = t

    fig = plt.figure()
    visualize_ray_casting(H, W, near_threshold,
                          far_threshold, num_samples, c2w, fig)
    plt.show()

if __name__ == '__main__':
    main()
