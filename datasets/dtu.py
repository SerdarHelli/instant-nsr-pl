import os
import json
import math
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, worldNormal):
    H,W,_ = worldNormal.shape
    normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def trans_normal(normal, RT_w2c, RT_w2c_target):

    normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    return normal_target_cam

def img2normal(img):
    return (img/255.)*2-1

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def norm_normalize(normal, dim=-1):

    normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

    return normal

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]


    return normal_cv

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
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

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
    up = rot_axis
    rot_dir = torch.cross(rot_axis, cam_center)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w

class DTUDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        dirs=os.listdir(os.path.join(self.config.root_dir, 'normal'))
        dirs=[x.split(".")[0] for x in dirs]

        img_sample = cv2.imread(os.path.join(self.config.root_dir, 'normal', dirs[0]+".png"))
        H, W = img_sample.shape[0], img_sample.shape[1]

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.factor = w / W

        mask_dir = os.path.join(self.config.root_dir, 'mask')
        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        self.directions = []
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        n_images =len(dirs)

        for i in range(n_images):
            cam_name=dirs[i].split("_")[1:]

            cameras_data=np.load(os.path.join(self.config.root_dir, 'camera', ("_").join(cam_name)+".npz"))
            K, c2w = cameras_data["K"],cameras_data["RT"]
            RT_cv=np.float32(RT_opengl2opencv(np.float32(c2w)))
            fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
            directions = get_ray_directions(w, h, fx, fy, cx, cy)
            self.directions.append(directions)

            c2w=inv_RT(RT_cv)
            c2w = torch.from_numpy(c2w).float()
            self.all_c2w.append(c2w)


            if self.split in ['train', 'val']:


                #img_path = os.path.join(self.config.root_dir, 'image', dirs[i]+".png")
                #img = np.array(Image.open(img_path).resize(self.img_wh, Image.BICUBIC))

                normal_path=os.path.join(self.config.root_dir, 'normal', dirs[i]+".png")
                normal = np.array(Image.open(normal_path).resize(self.img_wh, Image.BICUBIC))
                mask = np.array(Image.open(normal_path))
                mask[mask==127]=0
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
                mask=cv2.resize(mask,self.img_wh)
                normal = img2normal(normal)
                normal_cam_cv = normal_opengl2opencv(normal)
                normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)

                self.all_images.append(TF.to_tensor(normal2img(normal_world)).permute(1, 2, 0)[...,:3])
                self.all_fg_masks.append(TF.to_tensor(mask)[0]) # (h, w)

        self.all_c2w = torch.stack(self.all_c2w, dim=0)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.directions = self.directions[0]
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0), torch.stack(self.all_fg_masks, dim=0)
            self.directions = torch.stack(self.directions, dim=0)

        self.directions = self.directions.float().to(self.rank)
        self.all_c2w, self.all_images, self.all_fg_masks = \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_fg_masks.float().to(self.rank)



class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index
        }


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('dtu')
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DTUIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DTUDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = DTUDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = DTUDataset(self.config, 'train')

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
