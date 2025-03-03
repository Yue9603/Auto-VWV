import torch

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math



def dice_coef(y_true, y_pred, axis=(2, 3, 4)):
    smooth = 1.
    a = torch.sum(y_true * y_pred, axis)
    b = torch.sum(y_true**2, axis)
    c = torch.sum(y_pred**2, axis)
    dice = (2 * a + smooth) / (b + c + smooth)
    return torch.mean(dice)

def dice_loss(y_true, y_pred):
    d = dice_coef(y_true, y_pred)
    return 1 - d

def tdl_ca(y_pred, y_true):
    smooth = 1.
    y_true_vwv = torch.relu(y_true[:, :, 1:2] - y_true[:, :, 0:1])
    y_pred_vwv = torch.relu(y_pred[:, :, 1:2] - y_pred[:, :, 0:1])
    
    a = torch.sum(y_true * y_pred, (1, 3, 4)) 
    a_vwv = torch.sum(y_true_vwv * y_pred_vwv, (1, 3, 4))
    b = torch.sum(y_true, (1, 3, 4))
    b_vwv = torch.sum(y_true_vwv, (1, 3, 4))
    c = torch.sum(y_pred, (1, 3, 4))
    c_vwv = torch.sum(y_pred_vwv, (1, 3, 4))
    dice = (2 * a + smooth) / (b + c + smooth)
    dice_vwv = (2 * a_vwv + smooth) / (b_vwv + c_vwv + smooth)
    return 1 - (torch.mean(dice)*2 + torch.mean(dice_vwv)) / 3

def tdl_bce(y_pred, y_true):
    return (tdl_ca(y_pred, y_true) + B_crossentropy(y_pred, y_true)) / 2

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)

def prob_entropyloss(pred):
    pred = pred + 1e-5
    out = - pred * torch.log(pred)
    return torch.mean(out)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class FocalLoss:
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        # pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred + (1 - true) * (1 - pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        return loss.mean()
    

class DetLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, device, anchors, strides, a_box, a_obj, anchor_t=4.0, focal=False, gr=0, alpha=0.25, gamma=2):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        self.device = device  # get model device

        # Define criteria
        self.obj_loss_fn = B_crossentropy
        # Focal loss
        if focal:
            self.obj_loss_fn = FocalLoss(self.obj_loss_fn, gamma, alpha)

        self.gr = gr
        self.a_box, self.a_obj = a_box, a_obj
        self.anchor_t = anchor_t
        # self.stop_g = stop_g
        self.na = 1
        self.nc = 1  # number of classes
        self.nl = 1  # number of layers
        self.anchors = anchors
        self.strides = strides
        device

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        b, t, c, h, w = p.shape
        p = [torch.reshape(p, (b*t, 1, c, h, w)).permute(0, 1, 3, 4, 2)]
        tbox, indices, anchors, strides = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _ = pi[b, a, gj, gi].split((2, 2, 1), 1)  # target-subset of predictions

                # Regression
                pxy = pxy * 2 - 0.5
                pwh = (pwh * 2) ** 2 * anchors[i] / strides[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += self.obj_loss_fn(pi[...,4], tobj)

        loss = self.a_box * lbox + self.a_obj * lobj

        return loss

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tbox, indices, anch, stri = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets[None].repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape, stride = self.anchors[i], p[i].shape, self.strides[i]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(a,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None] * stride  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            stri.append(stride)  # anchors

        return tbox, indices, anch, stri
