import math
import numpy as np
import cv2
import torch

import warnings
warnings.filterwarnings('ignore')


# 该函数是用于生成距离矩阵的，其中将每个像素坐标之间在width方向的差值计算并求平方，再加上纵向差值的平方，最后再开根号即可得到距离矩阵，最后返回距离矩阵。
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
    return dist


# 是一个用于计算平均值和记录数据的 Python 类。常用于机器学习模型中的训练循环中，用来跟踪和记录损失函数、准确率等指标的变化情况。
class AverageMeter(object):     # 在训练循环中，可以使用 AverageMeter 类对象来记录每个 batch 的损失函数或准确率，然后汇总计算整个训练集的平均值。
    def __init__(self):
        self.history = None
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.clear()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 'nan'

    def new_epoch(self):
        self.history.append(self.avg)
        self.reset()


def get_iou(predict_masks, true_masks):# 获取iou指标
    n_samples, n_classes, H, W = predict_masks.size() # 获取预测结果和真实结果的形状


    prediction_max, prediction_argmax = predict_masks.max(-3) # 获取预测结果中最大值张量索引，返回最大值，及其通道
    prediction_argmax = prediction_argmax.long() # 将索引转化为整型

    classes = true_masks.new_tensor([c for c in range(n_classes)]).view(1, n_classes, 1, 1) # 初始化类别信息

    pred_bin = (prediction_argmax.view(n_samples, 1, H, W) == classes) # 预测结果二值化
    gt_bin = (true_masks.view(n_samples, 1, H, W) == classes) # 真实结果二值化


    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1) # 计算交集大小
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1) # 计算并集大小

    return (intersection + 1e-7) / (union + 1e-7) # 计算IoU


def db_eval_iou(annotation, segmentation):
    annotation = annotation.astype(np.bool_) # 将注释图像二值化
    segmentation = segmentation.astype(np.bool_) # 将分割结果二值化
    void_pixels = np.zeros_like(segmentation) # 初始化void像素
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1)) # 计算交集大小
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1)) # 计算并集大小
    j = inters / union # 计算IoU

    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j # 如果计算的是单个IoU，则判断是否存在并集为0的情况，如果是则将IoU设为1
    else:
        j[np.isclose(union, 0)] = 1 # 如果计算的是多个IoU，则将并集为0的IoU设为1

    return j # 返回IoU


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    if annotation.ndim == 3: # 判断是否为多帧图像
        n_frames = annotation.shape[0] # 获取图像序列的长度
        f_res = np.zeros(n_frames) # 初始化F-measure列表
        for frame_id in range(n_frames):
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], bound_th=bound_th) # 计算F-measure值并存入列表中
        return f_res # 返回F-measure列表
    elif annotation.ndim == 2: # 如果不是多帧图像
        f_res = f_measure(segmentation, annotation, bound_th=bound_th) # 计算F-measure值
        return f_res # 返回F-measure值


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """
    计算两个二值图像之间的 F 值。
    Args:
        foreground_mask: 一个二维的 bool 类型的数组，代表前景的二值图像。
        gt_mask: 一个二维的 bool 类型的数组，代表真实分割的二值图像。
        bound_th: 用来计算边界的阈值。默认为 0.008。
    Returns:
        F: 一个标量，代表 F 值。
    """
    # 创建一个与 foreground_mask 相同形状的 bool 数组，并将其所有元素设置为 False
    void_pixels = np.zeros_like(foreground_mask).astype(np.bool_)

    # 如果 bound_th < 1，则将其转换为大于等于 1 的整数，否则将其视为边界像素的数量
    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # 获取前景和真实分割的轮廓图
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    # 对前景和真实分割的轮廓图进行膨胀操作
    from skimage.morphology import disk
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # 获取前景和真实分割的匹配轮廓
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # 计算前景和真实分割的像素数量
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # 如果前景掩码中没有像素，但真实掩码中有，则精确度为 1，召回率为 0
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0

    # 如果前景掩码中有像素，但真实掩码中没有，则精确度为 0，召回率为 1
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    # 如果预测前景和真实前景都不存在，则将精确度和召回率设置为1。
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    # 否则，计算精确度和召回率。精确度表示预测为前景的区域中有多少是真正的前景区域。而召回率表示真实前景区域中有多少被成功预测出来了。
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    # 最后计算F分数（F-score），如果精确度和召回率同时为0，则F分数为0。
    if precision + recall == 0:
        F = 0
    else:   # 它是精确度（precision）和召回率（recall）的加权调和平均值。
        F = 2 * precision * recall / (precision + recall)
    return F


def _seg2bmap(seg, width=None, height=None):
    """
    将二值图像转换为轮廓图。
    Args:
        seg: 一个二维的 bool 类型的数组，代表输入的二值图像。
        width: 输出轮廓图的宽度。默认为 None。
        height: 输出轮廓图的高度。默认为 None。
    Returns:
        bmap: 一个二维的 bool 类型的数组，代表输出的轮廓图。
    """
    # 将输入的二值图像转换为 bool 类型
    seg = seg.astype(np.bool_)

    # 将非零元素变成 1
    seg[seg > 0] = 1

    # 如果没有指定宽度和高度，则使用 seg 图像的宽度和高度
    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    # 获取 seg 的宽度和高度
    h, w = seg.shape[:2]

    # 计算东、南、东南三个方向的邻居的位置
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    # 计算边界的位置
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    # 如果输出轮廓图的宽度和高度与输入不同，则进行插值
    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap # 返回二值数组，代表输出的轮廓图


def db_statistics(per_frame_values):  # 每帧的值
    with warnings.catch_warnings(): # 忽略 RuntimeWarning 警告
        warnings.simplefilter('ignore', category=RuntimeWarning)

        Metric = np.nanmean(per_frame_values)   # 计算每帧值的平均数
        O_ret = np.nanmean(per_frame_values > 0.5)  # 取值大于 0.5 的帧的占比的平均数    # nan取值为0且取均值时忽略它

    # 将数据分为 4 个区间，计算第一个和第四个区间的平均值之差
    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)
    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        D_ret = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])


    return Metric, O_ret, D_ret # 返回三个值：Metric、O_ret 和 D_ret


