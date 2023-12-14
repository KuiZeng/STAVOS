import cv2
import numpy as np
from PIL import Image

import torch
from train.train_data_deal import getEdgeClear

import warnings
warnings.filterwarnings('ignore')



def LabelToLongTensor(pic):
    # 如果输入是NumPy数组，则将其转换为PyTorch张量
    if isinstance(pic, np.ndarray):
        label = torch.from_numpy(pic).long()
    # 如果输入是单通道二值图像，则将其转换为PyTorch张量
    elif pic.mode == '1':
        label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
    else:
        # 如果输入是多通道图像，则将其转换为PyTorch张量
        label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # 如果输入是带透明通道的LA模式，则只保留亮度通道
        if pic.mode == 'LA':
            label = label.view(pic.size[1], pic.size[0], 2)
            label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
            label = label.view(1, label.size(0), label.size(1))
        else:
            # 如果输入是RGB或RGBA模式，则将所有通道并成一个
            label = label.view(pic.size[1], pic.size[0], -1)
            label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
    # 将标签中为255的像素值替换为1（该数据集中，255代表背景，1代表前景）
    # label[label == 255] = 1   #原本是0
    label[label == 255] = 1
    return label


def make_imgs_sta(img_path, mask_path, img_names, mask_names):
    frames = []
    max_clear = 0
    sta_frame_id = 0


    for i in range(len(img_names)):
        img = cv2.imread(img_path + img_names[i])
        mask  = cv2.imread(mask_path + mask_names[i], 0)
        clear = getEdgeClear(img, mask, 3)
        # clear = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if clear > max_clear:       # 如果当前帧的清晰度比之前的帧都高，更新关键帧
            max_clear = clear
            sta_frame_id = i

        img = img.astype(np.float32) / 255  # 归一化并转float32
        img = torch.tensor(img)  # 转tensor
        img = img.permute(2, 0, 1)  # (H, W, C)转(C, H, W)
        frames.append(img)  # (L, C, H, W)
    # print(sta_frame_id)
    imgs = torch.stack(frames).unsqueeze(0)  # (B, L, C, H, W)   B=1
    return imgs, sta_frame_id


def make_imgs(path, img_names):
    frames = []
    for img_name in img_names:
        img = cv2.imread(path + img_name)

        img = img.astype(np.float32) / 255  # 归一化并转float32
        img = torch.tensor(img)  # 转tensor
        img = img.permute(2, 0, 1)  # (H, W, C)转(C, H, W)
        frames.append(img)  # (L, C, H, W)
    imgs = torch.stack(frames).unsqueeze(0)  # (B, L, C, H, W)   B=1
    return imgs


def make_masks(path, mask_names):
    frames = []
    for mask_name in mask_names:
        mask = Image.open(path + mask_name)
        mask = LabelToLongTensor(mask)  # (C,H,W)
        frames.append(mask)  # (L, C, H, W)
    masks = torch.stack(frames).unsqueeze(0)  # (B, L, C, H, W)   B=1
    return masks


def make_given_masks(path, mask_names, sta_frame_id):
    frames = []
    mask = Image.open(path + mask_names[sta_frame_id])
    mask = LabelToLongTensor(mask)  # (C, H, W)

    sta_mask = mask.unsqueeze(0)  # (B, C, H, W)
    frames.append(sta_mask)
    frames.extend([None] * (len(mask_names) - 1))
    return frames
