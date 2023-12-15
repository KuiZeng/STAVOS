# -*- coding:utf-8 -*-
import numpy as np
import json
import cv2
import torch

import warnings
warnings.filterwarnings('ignore')


def videoLoad(video_name, out_path):  # 定义一个视频加载函数，参数：视频名，最大帧，灰度图像    # 函数返回参数为frames列表，视频fps
    video = cv2.VideoCapture("../input/" + video_name + ".avi")  # 获取视频
    fps = video.get(cv2.CAP_PROP_FPS)  # 获取视频每秒传输帧数

    frames = []  # 视频帧 列表
    i = 0

    sta_id = 0
    max_clear = 0
    sta = None
    ret, frame = video.read()  # videoCapture.read()按帧读取视频
    while ret:  # 当ret为True时(没读到末尾),继续读取
        cv2.imwrite(out_path + "frames/" + video_name + "_{:05d}.png".format(i), frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (480, 640)
        # print(gray.shape)
        clear = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 如果当前帧的清晰度比之前的帧都高，更新关键帧
        if clear > max_clear:
            max_clear = clear
            sta = frame.copy()
            sta_id = i

        frame = frame.astype(np.float32) / 255  # 归一化并转float32
        img = torch.from_numpy(frame).permute(2, 0, 1)  # (H, W, C)的numpy.ndarray或img转为(C, H, W)的tensor
        frames.append(img)
        i += 1
        ret, frame = video.read()  # videoCapture.read()按帧读取视频
    imgs = torch.stack(frames).unsqueeze(0)
    print("视频读取成功,sta帧为：", sta_id)

    cv2.imwrite(out_path + "sta.png", sta)
    return fps, imgs, sta_id


def json_to_mask(out_path, H, W):
    mask = np.zeros([H, W, 1], np.uint8)  # 创建一个大小和原图相同的纯黑画布

    with open(out_path + "sta.json", "r") as file:  # 读取json文件
        json_file = json.load(file)

    shapes = json_file["shapes"]
    for shape in shapes:  # shape是一个字典，有label,points点信息
        points = shape["points"]  # 获取json当前shape的点阵信息(list类型数组)
        # 填充
        points_array = np.array(points, dtype=np.int32)  # 列表数组转为numpy数组
        if shape["label"] == "ventricle":
            mask = cv2.fillPoly(mask, [points_array], 255)

    cv2.imwrite(out_path + "sta_mask.png", mask)  # 将mask对象保存到 masks文件夹下，文件名为img_mask.png
    print("最佳帧掩膜制作成功")


def sta_mask_load(mask):  # 加载最佳帧掩膜
    mask = mask[:, :, 0:1]
    mask = torch.tensor(mask)
    mask = mask.permute(2, 0, 1)
    mask[mask == 255] = 1
    mask = mask.unsqueeze(0)
    return mask
