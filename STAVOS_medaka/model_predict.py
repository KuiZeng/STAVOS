
import cv2
import torch

from STAVOS import STAVOS


def get_dist(H=384, W=384):
    h = (H + 15) // 16 # 将高和宽分别除以16向上取整，计算得到块的数量
    w = (W + 15) // 16
    dist = torch.zeros(h * w, h * w).cuda() # 初始化距离矩阵
    block = torch.zeros(w, w).cuda() # 初始化块矩阵
    for i in range(w):
        for j in range(w):
            block[i, j] = (i - j) ** 2 # 计算每个像素坐标之间在width方向的差值并取平方
    for i in range(h):
        for j in range(h):
            dist[i * w: (i + 1) * w, j * w: (j + 1) * w] = (block + (i - j) ** 2) ** 0.5 # 填充距离矩阵
    return dist     # torch.Size([1200, 1200])


def model_predict(model_name, imgs, given_masks, dist, sta_id):
    model = STAVOS().eval().cuda()
    model.load_state_dict(torch.load(model_name))

    with torch.no_grad():
        print("正在进行预测")
        vos_out = model(imgs, given_masks, dist.unsqueeze(0), None, sta_id, "test")

    return vos_out


def save_predict(vos_out, out_path, video_name):
    print("正在保存预测")
    predict_list = torch.squeeze(vos_out['masks'])  # torch.Size([1, 166, 1, 480, 640])  转 torch.Size([166, 480, 640])
    predict_list = predict_list.cpu().numpy()  # (166, 480, 640)
    for i, predict in enumerate(predict_list):
        predict[predict > 0.5] = 255
        cv2.imwrite(out_path + "predicts/" + video_name + "_{:05d}.png".format(i), predict)

