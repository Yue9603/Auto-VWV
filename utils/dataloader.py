import os
import SimpleITK as sitk
import cv2
from torch.utils import data
import numpy as np
import torch
import random
# import ants
from monai.transforms import RandGaussianNoise, RandBiasField, RandKSpaceSpikeNoise, RandGaussianSharpen, RandAdjustContrast
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

    
class Dataset_CA_CAPUNet(data.Dataset):
    def __init__(self, dir_):
        super(Dataset_CA_CAPUNet, self).__init__()
        self.filenames = dir_
        self.ns = len(self.filenames[0][0])

    def __getitem__(self, index):
        image_files = self.filenames[0][index]
        label_files = self.filenames[1][index]  # [lib, mab, cca, ica, eca]

        # t, c, h, w
        images = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in image_files], axis=0)[:, np.newaxis, ...]
        masks_mab = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files[1]], axis=0)
        masks_lib = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files[0]], axis=0)
        masks_mab_cca = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files[2]], axis=0)
        masks_mab_ica = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files[3]], axis=0)
        masks_mab_eca = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files[4]], axis=0)
        masks = np.stack((masks_lib, masks_mab, masks_mab_cca, masks_mab_ica, masks_mab_eca), axis=1)   # c=5, t, h, w

        return images, masks

    def __len__(self):
        return len(self.filenames[0])
    
def get_det_label(mask):
    """output (image,class,x,y,w,h)"""
    b, t, c, h, w = mask.shape
    bt_ = torch.linspace(0, b*t-1, b*t, dtype=mask.dtype, device=mask.device)
    c_ = torch.linspace(0, 2, 3, dtype=mask.dtype, device=mask.device)
    h_ = torch.linspace(0, h-1, h, dtype=mask.dtype, device=mask.device)
    w_ = torch.linspace(0, w-1, w, dtype=mask.dtype, device=mask.device)
    BT, C, Y, X = torch.meshgrid(bt_, c_, h_, w_)
    mask_mab = torch.reshape(mask[:, :, 2:], (-1, 3, h, w))
    y_ca = torch.where(mask_mab > 0, Y, torch.zeros_like(mask_mab)).view(b*t, 3, -1)
    x_ca = torch.where(mask_mab > 0, X, torch.zeros_like(mask_mab)).view(b*t, 3, -1)
    ca_x1 = torch.min(x_ca, dim=-1)[0] / w   # bt, 3
    ca_y1 = torch.min(y_ca, dim=-1)[0] / h
    ca_x2 = torch.max(x_ca, dim=-1)[0] / w
    ca_y2 = torch.max(y_ca, dim=-1)[0] / h
    
    center_x, center_y = (ca_x1 + ca_x2) / 2, (ca_y1 + ca_y2) / 2   # bt, 3
    w, h = ca_x2 - ca_x1, ca_y2 - ca_y1
    BT = BT[:, :, 0, 0]
    BT_mask = (BT+1) * w
    ti = BT[BT_mask > 0]
    ci = torch.zeros_like(ti)
    center_x = center_x[BT_mask > 0]
    center_y = center_y[BT_mask > 0]
    w = w[BT_mask > 0]
    h = h[BT_mask > 0]
    ci = torch.zeros_like(ti)
    
    target = torch.stack([ti, ci, center_x, center_y, w, h], dim=1)
    return target
    

def read_ieca_sep_mask_label(img_path, replace_str):
    ica_mab_mask_i = cv2.imread(img_path.replace(replace_str, 'mask_mab_ica'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    eca_mab_mask_i = cv2.imread(img_path.replace(replace_str, 'mask_mab_eca'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    ica_mab_mask_i = cv2.erode(ica_mab_mask_i, kernel, iterations=1)
    eca_mab_mask_i = cv2.erode(eca_mab_mask_i, kernel, iterations=1)
    return ica_mab_mask_i[..., np.newaxis], eca_mab_mask_i[..., np.newaxis]

def separate_pred_ieca_mask_manual(mask_ica, mask_eca):
    """pred_mask_cca2ieca [h, w, frames, 2]"""
    index_ica = np.stack(np.where(mask_ica[..., 0] > 0), 0)
    index_eca = np.stack(np.where(mask_eca[..., 0] > 0), 0)
    center_ica = np.mean(index_ica, 1, keepdims=True)
    center_eca = np.mean(index_eca, 1, keepdims=True)
    index_edge_eca = index_eca[:, np.argmin(np.linalg.norm(index_eca - center_ica, axis=0, keepdims=True), axis=1)]
    index_edge_ica = index_ica[:, np.argmin(np.linalg.norm(index_ica - index_edge_eca, axis=0, keepdims=True), axis=1)]
    # center_ieca = (index_edge_ica + index_edge_eca) / 2.
    center_ieca = (index_edge_ica + index_edge_eca).squeeze() / 2.
    # vect = (center_eca - center_ica) / np.linalg.norm(center_eca - center_ica, axis=0)
    vect = ((center_eca - center_ica) / np.linalg.norm(center_eca - center_ica, axis=0)).squeeze()
    mask_1 = np.zeros_like(mask_ica)
    mask_2 = np.zeros_like(mask_eca)
    if abs(vect[0])/abs(vect[1]) >= 1.0*mask_ica.shape[0] / mask_ica.shape[1]:
        cols = np.array(range(mask_ica.shape[1])).astype(np.float64)
        # rows = (cols - center_ieca[1]) / (-1) / vect[0] / vect[1] + center_ieca[0]
        rows = np.clip((cols - center_ieca[1]) / (-1*vect[0]) * vect[1] + center_ieca[0], 0, mask_ica.shape[0]-1)
        cols = cols[(0 <= rows) * (rows < mask_ica.shape[0])]
        rows = rows[(0 <= rows) * (rows < mask_ica.shape[0])]
        for i in range(len(cols)):
            mask_1[:int(np.round(rows[i])), i] = 1
            mask_2[int(np.round(rows[i])):, i] = 1
        if center_ica[0] > center_eca[0]:
            mask_ica_bool = mask_2
            mask_eca_bool = mask_1
        else:
            mask_ica_bool = mask_1
            mask_eca_bool = mask_2
    else:
        rows = np.array(range(mask_ica.shape[0])).astype(np.float64)
        # cols = (rows - center_ieca[0]) / (-1) / vect[1] * vect[0] + center_ieca[1]
        cols = np.clip((rows - center_ieca[0]) / (-1*vect[1]) * vect[0] + center_ieca[1], 0, mask_ica.shape[1]-1)
        cols = cols[(0 <= cols) * (cols < mask_ica.shape[1])]
        rows = rows[(0 <= cols) * (cols < mask_ica.shape[1])]
        for i in range(len(rows)):
            mask_1[i, :int(cols[i])] = 1
            mask_2[i, int(cols[i]):] = 1
        if center_ica[1] > center_eca[1]:
            mask_ica_bool = mask_2
            mask_eca_bool = mask_1
        else:
            mask_ica_bool = mask_1
            mask_eca_bool = mask_2
    # pred_mask_ica = pred_mask_ieca * mask_ica_bool
    # pred_mask_eca = pred_mask_ieca * mask_eca_bool

    # return pred_mask_ica, pred_mask_eca
    return mask_ica_bool

class Dataset_CAIN(data.Dataset):
    def __init__(self, dir_):
        super(Dataset_CAIN, self).__init__()
        self.filenames = dir_

    def __getitem__(self, index):
        image_files = self.filenames[0][index]
        label_files = self.filenames[1][index]

        # t, c, h, w
        images = []
        masks_mab = []
        masks_lib = []
        masks_bool = []
        for fim in image_files:
            images.append(cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
        for fma in label_files:
            masks_lib.append(cv2.imread(fma[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
            masks_mab.append(cv2.imread(fma[1], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
            ieca_flag = 'ica' in fma[0]
            cca_flag = 'cca' in fma[0]
            if ieca_flag:
                mask_ica_i, mask_eca_i = read_ieca_sep_mask_label(fma[1], "mask_mab_ica")
                mask_bool = separate_pred_ieca_mask_manual(mask_ica_i, mask_eca_i)
            else:
                mask_bool = np.ones_like(masks_lib[-1])
            masks_bool.append(mask_bool.squeeze())
        images = np.stack(images, axis=0)[:, np.newaxis, ...]
        masks_mab = np.stack(masks_mab, axis=0)
        masks_lib = np.stack(masks_lib, axis=0)
        masks_bool = np.stack(masks_bool, axis=0)
        masks = np.stack((masks_lib, masks_mab), axis=1)
        masks_bool = np.stack((masks_bool, masks_bool), axis=1)

        return images, masks, masks_bool

    def __len__(self):
        return len(self.filenames[0])
    

class Dataset_SPARC(data.Dataset):
    def __init__(self, dir_, **ignore_kwargs):
        super(Dataset_SPARC, self).__init__()
        self.filenames = dir_

    def __getitem__(self, index):
        image_files = self.filenames[0][index]
        label_files = self.filenames[1][index]

        # t, c, h, w
        images = np.stack([cv2.imread(fim, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in image_files], axis=0)[:, np.newaxis, ...]
        masks_mab = np.stack([cv2.imread(fim[1], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files], axis=0)
        masks_lib = np.stack([cv2.imread(fim[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for fim in label_files], axis=0)         
        masks = np.stack((masks_lib, masks_mab), axis=1)

        return images, masks

    def __len__(self):
        return len(self.filenames[0])
    
    
