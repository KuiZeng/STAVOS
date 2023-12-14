# -*- coding:utf-8 -*-
import os
import time
import numpy as np
import random
import torch.utils.data

from STAVOS import STAVOS
from train.train_data_load import TrainDAVIS
from test.test_data_load import TestData
from train.train_fit import model_fit

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    t0 = time.time()

    seed = 19970613
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 训练阶段,及参数设置
    model = STAVOS().eval().cuda()     # 模型
    model_path = "../trained_model/2016+2017+N+R"        #
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    t0 = time.time()
    # 路径             训练数据               随机帧列表长度  随机帧列表数量             验证数据               保存路径  权重名字   学习率  batch_size   训练步数,保存步数, 验证步数
    model_fit(model, TrainDAVIS('../DAVIS-2016-trainval/DAVIS',      '2016', 'train', clip_l=10, clip_n=128), TestData('../DAVIS-2016-trainval/DAVIS',      '2016', 'val'), model_path, "davis2016", 1e-4, 1, 400, 200, 10)
    model_fit(model, TrainDAVIS('../DAVIS-2017-trainval-480p/DAVIS', '2017', 'train', clip_l=10, clip_n=128), TestData('../DAVIS-2017-trainval-480p/DAVIS', '2017', 'val'), model_path, "davis2017", 1e-5, 1, 400, 200, 10)

    model_fit(model, TrainDAVIS('../Data-medaka-Lateral-right/better', '2023', 'train', clip_l=10, clip_n=128), TestData('../Data-medaka-Lateral-right/better', '2023', 'val'), model_path, "N_better", 1e-5, 1, 400, 400, 10)
    model_fit(model, TrainDAVIS('../Data-medaka-Ventral/better',       '2023', 'train', clip_l=10, clip_n=128), TestData('../Data-medaka-Ventral/better',       '2023', 'val'), model_path, "R_better", 1e-5, 1, 400, 400, 10)

    # model_fit(model, TrainDAVIS('../Data-medaka-Lateral-right/lower', '2023', 'train', clip_l=10, clip_n=32), TestData('../Data-medaka-Lateral-right/lower', '2023', 'val'), model_path, "N_lower", 1e-5, 1, 400, 400, 10)
    # model_fit(model, TrainDAVIS('../Data-medaka-Ventral/lower', '2023', 'train', clip_l=10, clip_n=32), TestData('../Data-medaka-Ventral/lower', '2023', 'val'), model_path, "R_lower", 1e-5, 1, 400, 50, 10)

    t1 = time.time()
    print("全部训练完成！总共花费了时间：", t1 - t0)
