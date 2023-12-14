# -*- coding:utf-8 -*-
import json
import time
import copy
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
from train.train_data_deal import getEdgeClear
from train.train_metric import AverageMeter, get_iou, db_eval_iou, db_eval_boundary, db_statistics, get_dist

import warnings
warnings.filterwarnings('ignore')


def model_fit(model, train_set, val_set, model_path, save_name, learning_rate, batch_size, epochs, save_step, val_step):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    trainer = Trainer(model, optimizer, train_loader, val_set, model_path=model_path, save_name=save_name, save_step=save_step, val_step=val_step)
    trainer.train(epochs)
    trainer.print_plt()



class Trainer(object):  # 模型，优化器，训练数据加载器，验证集，保存名字，保存间隔，验证间隔
    def __init__(self, model, optimizer, train_loader, val_set, model_path, save_name, save_step, val_step):
        self.model = model.cuda()  # 模型
        self.optimizer = optimizer  # 优化器
        self.train_loader = train_loader  # 训练数据加载器
        self.val_set = val_set  # 验证集

        self.model_path = model_path
        self.save_name = save_name  # 保存名字
        self.save_step = save_step  # 保存间隔
        self.val_step = val_step    # 验证间隔
        self.epoch = 1              # 当前训练轮次
        self.best_score = 0         # 最好分数
        self.score = 0              # 当前分数
        self.stats = {'loss': AverageMeter(), 'iou': AverageMeter()}  # 损失函数，IOU指标
        self.dist = get_dist()  # 距离矩阵

        self.loss_list = []
        self.iou_list = []
        self.JF_list = []
        self.t0 = time.time()

    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()

            if self.epoch % self.save_step == 0:  # 每过save_step保存一次
                self.save_checkpoint()  # 保存模型权重参数

            if self.score > self.best_score:  # 如果最优，自动保存
                print('-------------------------------------在 epoch{} 后, 出现了新的最优模型权重参数--------------------------\n'.format(self.epoch))
                self.save_checkpoint(alt_name='best')  # 保存模型权重参数
                self.best_score = self.score
        print('训练完成!\n')

    def train_epoch(self):

        # 训练
        t1 = time.time()
        self.cycle_dataset(mode='train')
        t2 = time.time()
        print("该epoch花费时间:", t2 - t1, "  该数据集累计花费时间:", t2 - self.t0)


        # 验证
        if self.epoch % self.val_step == 0:
            if self.val_set is not None:
                with torch.no_grad():
                    t1 = time.time()
                    self.score = self.cycle_dataset(mode='val')
                    t2 = time.time()
                    print("该epoch花费时间:", t2 - t1, "  该数据集累计花费时间:", t2 - self.t0)

        # 更新统计信息（即描述数据集特征的数值）
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for dataset_data in self.train_loader:
                imgs = dataset_data['imgs'].cuda()
                masks = dataset_data['masks'].cuda()
                B, L, _, H, W = imgs.size()  # torch.Size([2, 10, 3, 384, 384])
                # print(masks.size())
                # 《交换附加增强》
                if random.random() > 0.8:
                    objects = torch.roll(imgs * masks, dims=0, shifts=1)
                    object_masks = torch.roll(masks, dims=0, shifts=1)
                    imgs = (1 - object_masks) * imgs + object_masks * objects
                    masks = (1 - object_masks) * masks

                # sta帧
                sta_frame_id = 0
                if random.random() > 0.5:
                    max_clear = 0
                    img_frames = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()*255  # 变成B, L, H, W, C
                    mask_frames = masks.permute(0, 1, 3, 4, 2).cpu().numpy()*255
                    for i in range(L):
                        clear = getEdgeClear(img_frames[0, i, :, :, :], mask_frames[0, i, :, :, 0], 3)
                        if clear > max_clear:
                            max_clear = clear
                            sta_frame_id = i
                            # sta = cv2.cvtColor(img_frames[0, i, :, :, :].numpy()*255, cv2.COLOR_BGR2RGB)
                            # cv2.imwrite("sta.png", sta)


                given_masks = [masks[:, sta_frame_id]] + (L - 1) * [None]  # 放入sta帧

                # 如果sta帧没有目标，跳过这个批次
                skip = False
                for batch in range(B):
                    # print("进入模型前", masks[batch, sta_frame_id].unique().tolist())
                    if masks[batch, sta_frame_id].unique().tolist() == [0]:
                        skip = True
                        break
                if skip:
                    # print("跳过")
                    continue

                # 运行模型
                vos_out = self.model(imgs, given_masks, self.dist.unsqueeze(0).repeat(B, 1, 1), None, sta_frame_id, mode)  # print(vos_out['scores'].shape)    # torch.Size([2, 9, 2, 384, 384])
                # 真实掩膜与预测掩膜
                true_masks = torch.cat((masks[:, :sta_frame_id], masks[:, sta_frame_id + 1:]), dim=1).reshape(B * (L - 1), H, W)  # 除去sta帧 # ([18, 384, 384])
                predict_masks = vos_out['scores'].view(B * (L - 1), 2, H, W)  # torch.Size([18, 2, 384, 384])

                # 计算loss损失
                loss = nn.CrossEntropyLoss()(predict_masks, true_masks)  # torch.Size([18, 2, 384, 384])     torch.Size([18, 384, 384])

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算IOU指标
                self.stats['loss'].update(loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(predict_masks, true_masks))
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            self.loss_list.append(self.stats['loss'].avg)
            self.iou_list.append(self.stats['iou'].avg)
            print('epoch:{:04d}                  loss: {:.5f}                  iou: {:.5f}'.format(self.epoch, self.stats['loss'].avg, self.stats['iou'].avg))

        if mode == 'val':
            metrics_res = {'J': [], 'F': []}
            # print("正在进行验证")
            for video_name, dataset_data in self.val_set.data_load():  # 循环读取验证集
                # print("正在使用{}视频进行验证*****************************".format(video_name))

                imgs = dataset_data['imgs'].cuda()
                given_masks = [dataset_data['given_masks'][0].cuda()] + dataset_data['given_masks'][1:]
                masks = dataset_data['masks'].cuda()
                sta_frame_id = dataset_data["sta_frame_id"]

                # 距离矩阵
                dist = get_dist(imgs.size(-2), imgs.size(-1))

                # 推理
                vos_out = self.model(imgs, given_masks, dist.unsqueeze(0), None, sta_frame_id, mode)  # ([1, 71, 1, 480, 854])   # B,L,C,H,W

                # predict_masks = torch.cat( (vos_out['masks'][:, :sta_frame_id], vos_out['masks'][:, sta_frame_id+1:]), dim=1).squeeze(2)  # 预测掩膜      # ([1, 70, 480, 854])
                # true_masks = torch.cat((masks[:, :sta_frame_id], masks[:, sta_frame_id + 1:]), dim=1).squeeze(2)  # 真实掩膜      # ([1, 70, 480, 854])

                predict_masks = vos_out['masks'].squeeze(2)
                true_masks = masks.squeeze(2)

                # print(predict_masks.shape)
                # print(true_masks.shape)

                B, L, H, W = predict_masks.shape

                object_ids = list(map(int, np.unique(true_masks.cpu())))  # [0, 1,....]        # 对象id,只有一个对象的话，就是背景0和目标1
                object_ids.remove(0)  # 去掉前景

                # 评价预测结果
                all_predicts = np.zeros((len(object_ids), L, H, W))  # 初始化
                all_masks = np.zeros((len(object_ids), L, H, W))  # 初始化
                for i in object_ids:  # 遍历每个对象
                    predicts_i = copy.deepcopy(predict_masks).cpu().numpy()
                    predicts_i[predicts_i != i] = 0
                    predicts_i[predicts_i != 0] = 1
                    all_predicts[i - 1] = predicts_i[0]  # predicts_i[0]

                    masks_i = copy.deepcopy(true_masks).cpu().numpy()
                    masks_i[masks_i != i] = 0
                    masks_i[masks_i != 0] = 1
                    all_masks[i - 1] = masks_i[0]  # masks_i[0]

                # 计算J&F分数   # 其中J描述的是预测的mask和gt之间的IOU，  F描述的是预测mask边界与gt边界之间的吻合程度
                j_metrics_res = np.zeros(all_masks.shape[:2])  # 初始化
                f_metrics_res = np.zeros(all_masks.shape[:2])  # 初始化

                for ii in range(all_masks.shape[0]):  # len(object_ids)
                    j_metrics_res[ii] = db_eval_iou(all_masks[ii], all_predicts[ii])
                    [JM, _, _] = db_statistics(j_metrics_res[ii])
                    metrics_res['J'].append(JM)

                    f_metrics_res[ii] = db_eval_boundary(all_masks[ii], all_predicts[ii])
                    [FM, _, _] = db_statistics(f_metrics_res[ii])
                    metrics_res['F'].append(FM)

            # 聚合分数
            J, F = metrics_res['J'], metrics_res['F']
            final_mean = (np.mean(J) + np.mean(F)) / 2.
            self.JF_list.append(final_mean)
            print('**************************epoch:{:04d}                                 J&F score: {:.5f}************************\n'.format(self.epoch, final_mean))
            return final_mean

    def print_plt(self):

        json_data = {
            "loss_list": self.loss_list,
            "iou_list": self.iou_list,
            "JF_list": self.JF_list
        }

        with open(self.model_path + "/" + self.save_name + "训练过程数据.json", "w") as json_file:
            # 将json数据写入json文件
            json.dump(json_data, json_file)

        print("\"loss_list\":", self.loss_list, ",")
        print("\"iou_list\":", self.iou_list, ",")
        print("\"JF_list\":", self.JF_list, ",")

        plt.figure(figsize=(12, 8))
        plt.title('Loss, IoU and J&F')
        plt.xlabel('epochs')  # x轴名称
        plt.ylabel('metric')  # y轴名称

        plt.plot(np.arange(1, len(self.loss_list)+1), self.loss_list, 'b:', label='Loss')
        plt.plot(np.arange(1, len(self.loss_list)+1), self.iou_list, 'r', label='IoU')
        plt.plot(np.arange(10, len(self.loss_list)+1, 10), self.JF_list, 'g', label='J&F')

        plt.legend()
        plt.savefig(self.model_path + "/" + self.save_name + "训练过程折线图.png")

        print("训练过程折线图保存成功")

    def save_checkpoint(self, alt_name=None):  # 保存模型权重参数
        if alt_name is not None:
            file_path = self.model_path + '/{}_{}.pth'.format(self.save_name, alt_name)
        else:
            file_path = self.model_path + '/{}_{:04d}.pth'.format(self.save_name, self.epoch)
        torch.save(self.model.state_dict(), file_path)  # 字典形式保存

    def load_checkpoint(self, mode, epoch):  # 加载预训练模型
        checkpoint_path = self.model_path + '/{}_{:04d}.pth'.format(mode, epoch)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.epoch = epoch + 1
        print('loaded: {}'.format(checkpoint_path))
