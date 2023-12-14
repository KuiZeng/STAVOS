import os
import random
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF
import torch.utils.data

from test.test_data_deal import LabelToLongTensor
from train.train_data_deal import load_image_in_PIL, random_affine_params, random_crop_params

import warnings
warnings.filterwarnings('ignore')


class TrainDAVIS(torch.utils.data.Dataset):  # 1，随机选择视频列表中的视频，2，随机选择翻转视频帧序列，3，随机取其中连续10帧，4，随机选择上下/左右翻转视频帧，5，随机仿射变换，6，随机平衡裁剪
    def __init__(self, root, year, split, clip_l, clip_n):
        self.root = root
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_l = clip_l  # 列表长度
        self.clip_n = clip_n  # 一次选多少个列表
        self.to_tensor = tv.transforms.ToTensor()
        self.id = 0
        self.keyId = 0

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)  # 随机选择视频
        # print("正在加载{}视频".format(video_name))

        img_dir = self.root + "/JPEGImages/480p/" + video_name + "/"
        mask_dir = self.root + "/Annotations/480p/" + video_name + "/"
        img_frames = os.listdir(img_dir)
        mask_frames = os.listdir(mask_dir)

        # 50%概率随机翻转视频   即翻转视频前后顺序
        if random.random() > 0.5:
            img_frames.reverse()
            mask_frames.reverse()

        # 取clip_l长度的随机帧
        size = len(img_frames)
        random_number = random.randint(0, size-self.clip_l)  # 随机帧起点
        random_frames = [num for num in range(random_number, random_number + self.clip_l)]

        # print(size, random_frames)

        # 50%概率随机翻转帧    水平左右
        h_flip = False
        if random.random() > 0.5:
            h_flip = True  # horizontal
        v_flip = False
        if random.random() > 0.5:
            v_flip = True  # vertical

        # 生成训练片段
        img_list = []
        mask_list = []

        for i, frame_id in enumerate(random_frames):
            img = load_image_in_PIL(img_dir + img_frames[frame_id], 'RGB')
            mask = load_image_in_PIL(mask_dir + mask_frames[frame_id], 'P')

            # 加入翻转
            if h_flip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if v_flip:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # 随机仿射变换
            ret = random_affine_params(degree=5, translate=0, scale_ranges=(0.95, 1.05), shear=5, img_size=img.size)
            if i == 0:
                img = TF.affine(img, *ret, Image.BICUBIC)
                mask = TF.affine(mask, *ret, Image.NEAREST)
                old_ret = ret
            else:
                new_ret = (ret[0] + old_ret[0], (0, 0), ret[2] * old_ret[2], (ret[3][0] + old_ret[3][0], ret[3][1] + old_ret[3][1]))
                img = TF.affine(img, *new_ret, Image.BICUBIC)
                mask = TF.affine(mask, *new_ret, Image.NEAREST)
                old_ret = new_ret

            # 加入平衡随机裁剪
            # y, x, h, w = random_crop_params(mask, scale=(0.8, 1.25))  # 0 107 480 640
            if i == 0:
                for count in range(self.clip_l):
                    y, x, h, w = random_crop_params(mask, scale=(0.8, 1.25))  # 0 107 480 640
                    temp_mask = LabelToLongTensor(TF.resized_crop(mask, y, x, h, w, [384, 384], TF.InterpolationMode.NEAREST))

                    # 从参考系中选择一个对象
                    selected_id = 19970613
                    possible_obj_ids = temp_mask.unique().tolist()
                    if 0 in possible_obj_ids:
                        possible_obj_ids.remove(0)
                    if len(possible_obj_ids) > 0:
                        selected_id = random.choice(possible_obj_ids)
                    temp_mask[temp_mask != selected_id] = 0
                    temp_mask[temp_mask != 0] = 1

                    # 确保至少256个前景像素
                    if len(temp_mask[temp_mask != 0]) >= 256 or count == self.clip_l - 1:
                        img_list.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, [384, 384], TF.InterpolationMode.BICUBIC)))  # Image.BICUBIC
                        mask_list.append(temp_mask)
                        break
            else:
                img_list.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, [384, 384], TF.InterpolationMode.BICUBIC)))  #
                temp_mask = LabelToLongTensor(TF.resized_crop(mask, y, x, h, w, [384, 384], TF.InterpolationMode.NEAREST))
                temp_mask[temp_mask != selected_id] = 0
                temp_mask[temp_mask != 0] = 1
                mask_list.append(temp_mask)


        # 聚合所有帧
        imgs = torch.stack(img_list, 0)  # [10, 3, 384, 384]         # 【随机选择的帧数, C, H, W】
        masks = torch.stack(mask_list, 0)  # [10, 1, 384, 384]
        return {'imgs': imgs, 'masks': masks}

