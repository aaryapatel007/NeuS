import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os, json
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.image_dir = conf.get_string('image_dir')
        self.mask_dir = conf.get_string('mask_dir')
        self.normal_dir = conf.get_string('normal_dir')
        self.normal_mask_dir = conf.get_string('normal_mask_dir')
        self.render_cameras_name = conf.get_string('camera_params')
        self.n_views = conf.get_string('n_views')
        self.select_views = conf.get_string('select_views', default='')

        if(self.data_dir.split('/')[-1] in ['armadillo', 'bunny']):
            self.images_list = sorted(glob(os.path.join(self.image_dir, 'avg_l96', '*.png')))
        else:
            self.images_list = sorted(glob(os.path.join("".join([self.image_dir, "_intnorm_gt/avg"]), '*.png')))
            # self.images_list = []
            # directories = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]

            # # Iterate through each directory
            # for directory in directories:
            #     # Construct the full path to the current directory
            #     current_directory = os.path.join(self.image_dir, directory)
            #     if os.path.isfile(os.path.join(current_directory, self.image_num + '.png')):
            #         self.images_list.append(os.path.join(current_directory, self.image_num + '.png'))

        self.masks_lis = sorted(glob(os.path.join(self.mask_dir, '*.png')))
        self.normals_list = sorted(glob(os.path.join(self.normal_dir, '*.npy')))
        self.normal_masks_list = sorted(glob(os.path.join(self.normal_mask_dir, '*.png')))

        if self.n_views != 'all':
            # self.select_views = [int(x) - 1 for x in self.select_views.split()]

            # self.images_list = [self.images_list[i] for i in self.select_views]
            # self.masks_lis = [self.masks_lis[i] for i in self.select_views]
            # self.normals_list = [self.normals_list[i] for i in self.select_views]
            # self.normal_masks_list = [self.normal_masks_list[i] for i in self.select_views]

            # select n_views from the list of images by dividing the total number of images by n_views
            n_selected_images = len(self.images_list) // self.n_views

            if(n_selected_images > 1):
                self.images_list = self.images_list[::n_selected_images]
                self.masks_lis = self.masks_lis[::n_selected_images]
                self.normals_list = self.normals_list[::n_selected_images]
                self.normal_masks_list = self.normal_masks_list[::n_selected_images]
            else:
                self.images_list = self.images_list[:self.n_views]
                self.masks_lis = self.masks_lis[:self.n_views]
                self.normals_list = self.normals_list[:self.n_views]
                self.normal_masks_list = self.normal_masks_list[:self.n_views]

        self.n_images = len(self.images_list)

        print('Number of images: ', self.n_images)

        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_list]) / 255.0
        self.normal_vecs = np.stack([np.load(normal_file) for normal_file in self.normals_list])
        self.normal_masks_imgs = np.stack([cv.imread(im_name) for im_name in self.normal_masks_list]) / 255.0
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0

        with open(os.path.join(self.data_dir, self.render_cameras_name)) as file:
            camera_calib_dict = json.load(file)

        self.gt_normal_world = camera_calib_dict['gt_normal_world']

        K = np.asarray(camera_calib_dict['K'])

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        self.intrinsics_all = np.tile(intrinsics, (self.n_images, 1)).reshape(self.n_images, intrinsics.shape[0], intrinsics.shape[1])

        if(self.n_views != 'all'):
            # self.pose_all = np.asarray(camera_calib_dict['pose_c2w']).astype(np.float32)[self.select_views]
            
            if(n_selected_images > 1):
                self.pose_all = np.asarray(camera_calib_dict['pose_c2w']).astype(np.float32)[::n_selected_images]
            else:
                self.pose_all = np.asarray(camera_calib_dict['pose_c2w']).astype(np.float32)[:self.n_views]
        else:
            self.pose_all = np.asarray(camera_calib_dict['pose_c2w']).astype(np.float32)
        

        if(not self.gt_normal_world):
            self.normal_vecs = np.einsum('bij,bklj->bkli', self.pose_all[:, :3, :3], self.normal_vecs)
        
        # OpenGL to OpenCV coordinate system
        self.pose_all[:,:3,1:3] *= -1.

        # normalize normal vectors
        self.normal_vecs = self.normal_vecs / (np.linalg.norm(self.normal_vecs, axis=-1, keepdims=True) + 1e-8)

        # self.scale_mats_np = np.tile(np.eye(4), (self.n_images, 1)).reshape(self.n_images, 4, 4)
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.normals = torch.from_numpy(self.normal_vecs.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.normal_masks = torch.from_numpy(self.normal_masks_imgs.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        # dummy mask since no mask in the dataset
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]

        self.intrinsics_all = torch.from_numpy(self.intrinsics_all).float().to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.pose_all = torch.from_numpy(self.pose_all).float().to(self.device)  # [n_images, 4, 4]

        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.eye(4)
        object_bbox_min = np.linalg.inv(object_scale_mat) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(object_scale_mat) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), self.pose_all[img_idx], self.intrinsics_all[img_idx]
    
    def gen_rays_at_psnerf(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W, (self.W // l) + 1)[:-1]
        ty = torch.linspace(0, self.H, (self.H // l) + 1)[:-1]
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.stack([pixels_x, pixels_y], dim=-1) # W, H, 2
        p_trans = (p - self.intrinsics_all[img_idx,:2,2]) / self.intrinsics_all[img_idx,0,0]
        p_trans = torch.cat([p_trans, torch.ones_like(p_trans[...,:1])], dim=-1) 
        p_trans = p_trans / torch.linalg.norm(p_trans, ord=2, dim=-1, keepdim=True)
        # rays_v = torch.einsum('bij,bnj->bni',self.pose_all[img_idx, None, None, :3, :3], p_trans).squeeze(0)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], p_trans[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), self.pose_all[img_idx], self.intrinsics_all[img_idx]
    
    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device = self.device)
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device = self.device)
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_random_rays_at_psnerf(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device = self.device)
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device = self.device)

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        normal = self.normals[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        normal_mask = self.normal_masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        # # pixel to camera coordinate transformation
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3

        # # camera to world coordinate transformation
        # rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3

        p = torch.stack([pixels_x, pixels_y], dim=-1) # W, H, 2
        p_trans = (p - self.intrinsics_all[img_idx,:2,2]) / self.intrinsics_all[img_idx,0,0]
        p_trans = torch.cat([p_trans, torch.ones_like(p_trans[...,:1])], dim=-1) 
        rays_v = torch.einsum('bij,bnj->bni',self.pose_all[img_idx, None, :3, :3], p_trans.unsqueeze(0)).squeeze(0)
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)

        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color.cpu(), normal.cpu(), mask[:, :1].cpu(), normal_mask[:, :1].cpu()], dim=-1).cuda(), self.pose_all[img_idx], self.intrinsics_all[img_idx]  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_list[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)