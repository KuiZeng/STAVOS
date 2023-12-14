# -*- coding:utf-8 -*-
import os
import torch.utils.data

from test.test_data_deal import make_imgs_sta, make_given_masks, make_masks, make_imgs

import warnings
warnings.filterwarnings('ignore')


# class TestData(torch.utils.data.Dataset)
class TestData(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.video_list = sorted(f.read().splitlines())
            print('--- {} {} {} loaded ---'.format(self.root, self.year, self.split))

    def data_load(self):
        for video_name in self.video_list:
            # print("正在处理对{}视频进行数据集加载:".format(video_name))
            img_path = self.root + '/JPEGImages/480p/' + video_name + "/"
            mask_path = self.root + '/Annotations/480p/' + video_name + "/"
            img_names = os.listdir(img_path)
            mask_names = os.listdir(mask_path)

            # if "Davis" in self.root:
            #     imgs = make_imgs(img_path, img_names)
            #     sta_frame_id = 0
            # else:
            #     imgs, sta_frame_id = make_imgs_sta(img_path, mask_path, img_names, mask_names)
            imgs, sta_frame_id = make_imgs_sta(img_path, mask_path, img_names, mask_names)
            given_masks = make_given_masks(mask_path, mask_names, sta_frame_id)

            if 'test' in self.split:  # 如果是测试集
                yield video_name, {'imgs': imgs, 'given_masks': given_masks,                'val_frame_ids': None, "sta_frame_id": sta_frame_id}
            else:
                masks = make_masks(mask_path, mask_names)
                if self.year != '2017':
                    masks = (masks != 0).long()
                    given_masks[0] = (given_masks[0] != 0).long()
                # if self.year == '2016':
                #     masks = (masks != 0).long()
                #     given_masks[0] = (given_masks[0] != 0).long()
                yield video_name, {'imgs': imgs, 'given_masks': given_masks, 'masks': masks, 'val_frame_ids': None, "sta_frame_id": sta_frame_id}
