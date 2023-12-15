# -*- coding:utf-8 -*-
import numpy as np
import os
import random
import torch

from STAVOS import STAVOS
from test.test_data_load import TestData
from test.test_metric import get_metric_json
from test.test_predict import model_predict

import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    seed = 19970613
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = STAVOS().eval().cuda()
    model_path = "../trained_model/2016+2017+N+R"
    model_name = '/R_better_0400.pth'
    model.load_state_dict(torch.load(model_path + model_name))
    datasets = {

        # 'Davis2016_test': TestData('../DAVIS-2016-trainval/Davis', '2016', 'val'),
        'N_better': TestData('../Data-medaka-Lateral-right/better', '2023', 'test'),
        'R_better': TestData('../Data-medaka-Ventral/better', '2023', 'test'),

    }

    with torch.no_grad():
        for dataset_name, dataset in datasets.items():
            model_predict(model, dataset, "../test_predict/" + dataset_name)




