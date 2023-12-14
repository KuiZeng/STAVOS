# -*- coding:utf-8 -*-
import os
import time
import cv2
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


def resize_480p(H, W, imgs, given_masks):
    original_imgs = imgs.clone()
    original_given_masks = given_masks.copy()

    # resize to 480p
    resize = False

    if H > W:
        if W != 480:
            resize = True
            ratio = 480 / W
            imgs = F.interpolate(imgs[0], size=(int(ratio * H), 480), mode='bicubic', align_corners=False).unsqueeze(0)
            for i in range(len(given_masks)):
                if given_masks[i] is None:
                    continue
                else:
                    given_masks[i] = F.interpolate(given_masks[i].float(), size=(int(ratio * H), 480), mode='nearest').long()
    else:
        if H != 480:
            resize = True
            ratio = 480 / H
            imgs = F.interpolate(imgs[0], size=(480, int(ratio * W)), mode='bicubic', align_corners=False).unsqueeze(0)
            for i in range(len(given_masks)):
                if given_masks[i] is None:
                    continue
                else:
                    given_masks[i] = F.interpolate(given_masks[i].float(), size=(480, int(ratio * W)), mode='nearest').long()

    # back to original size if objects are too small
    if resize:
        tiny_obj = 0
        for i in range(len(given_masks)):
            if given_masks[i] is None:
                continue
            else:
                object_ids = given_masks[i].unique().tolist()
                if 0 in object_ids:
                    object_ids.remove(0)
                for obj_idx in object_ids:
                    if len(given_masks[i][given_masks[i] == obj_idx]) < 1000:
                        tiny_obj += 1
        if tiny_obj > 0:
            imgs = original_imgs
            given_masks = original_given_masks
    return imgs, given_masks


def getdist(H, W):
    h = (H + 15) // 16
    w = (W + 15) // 16
    dist = torch.zeros(h * w, h * w).cuda()
    block = torch.zeros(w, w).cuda()
    for i in range(w):
        for j in range(w):
            block[i, j] = (i - j) ** 2
    for i in range(h):
        for j in range(h):
            dist[i * w: (i + 1) * w, j * w: (j + 1) * w] = (block + (i - j) ** 2) ** 0.5

    return dist.unsqueeze(0)


def predict_save(vos_out, output_path, video_name):
    predict_list = torch.squeeze(vos_out['masks'])  # torch.Size([1, 166, 1, 480, 640])  转 torch.Size([166, 480, 640])
    predict_list = predict_list.cpu().numpy()  # (166, 480, 640)
    path = output_path + "/" + video_name + "/"
    # print(path + video_name + "_00001.png")
    for i, predict in enumerate(predict_list):
        predict[predict > 0.5] = 255            # 分割阈值
        cv2.imwrite(path + video_name + "_{:05d}.png".format(i), predict)


def model_predict(model, dataset, output_path):
    model.cuda()
    total_time, total_frames = 0, 0
    for video_name, dataset_data in dataset.data_load():    # 一次一个视频
        os.makedirs(output_path + "/" + video_name + "/", exist_ok=True)        # 创建输出目录文件夹

        imgs = dataset_data['imgs'].cuda()
        given_masks = [dataset_data['given_masks'][0].cuda()] + dataset_data['given_masks'][1:]
        val_frame_ids = dataset_data['val_frame_ids']
        sta_id = dataset_data["sta_frame_id"]

        B, L, C, H, W = imgs.size()
        imgs, given_masks = resize_480p(H, W, imgs, given_masks)
        dist = getdist(H, W)



        # print("正在对{}视频进行分割......".format(video_name), sta_id)
        t0 = time.time()
        vos_out = model(imgs, given_masks, dist, val_frame_ids, sta_id, "test")
        t1 = time.time()

        time_elapsed = t1 - t0
        print('{}分割结束,{:.1f}帧/秒'.format(video_name, L / time_elapsed))

        # print("正在{}视频的分割结果进行保存......".format(video_name))
        predict_save(vos_out, output_path, video_name)

        total_time += time_elapsed
        total_frames = total_frames + L
    print('该数据集全部处理结束 {:.1f}帧/秒'.format(total_frames / total_time))
