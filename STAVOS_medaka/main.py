# -*- coding:utf-8 -*-
import os
import time
import cv2

from data_input import videoLoad, json_to_mask, sta_mask_load
from model_predict import get_dist, model_predict, save_predict
from data_output import create_all_dir, save_3D, write_to_csv, save_electrocardiogram, save_visualization
from image_quantify import predict_deal, segment_analysis
from heart_quantify import get_systole_frames, get_diastole_frames, get_ventricular_dimensions

import warnings
warnings.filterwarnings('ignore')


def main():
    print("请输入要使用的模型")
    model_name = "../trained_model/" + input()
    # model_name = "../trained_model/R_exe_0300.pth"
    print("请输入要执行的视频名字")        # N0088
    video_name = input()  # 视频必须在../input/下
    # video_name = "R0039"        # R0039   N0080

    # 检查是否存在该视频
    file_exists = os.path.exists("../input/" + video_name + ".avi")
    while not file_exists:
        print("../input/" + video_name + ".avi", "不存在!!!")
        print("请输入正确的要执行的视频名...")   # N0068 N0080 N0094
        video_name = input()
        file_exists = os.path.exists("../input/" + video_name + ".avi")
    #
    # # 创建所有需要的文件夹目录
    out_path = create_all_dir("../output/" + video_name + "/")  # 如果不存在,则新建文件夹      如果存在,则新建非重名文件夹
    fps, imgs, sta_id = videoLoad(video_name, out_path)
    _, L, _, H, W = imgs.shape

    # 检查是否制作好最佳帧标签
    file_exists = os.path.exists((out_path + "sta.json"))
    while not file_exists:
        print(out_path + "sta.json", "不存在!!!请制作好sta帧(label为：ventricle)的标签后，输入yes继续...")
        time.sleep(10)
        while input() != "yes":
            print("输入yes继续")
        file_exists = os.path.exists((out_path + "sta.json"))

    # 制作sta帧掩膜
    json_to_mask(out_path, H, W)
    mask = cv2.imread(out_path + "sta_mask.png")
    sta_mask = sta_mask_load(mask)

    # 加载数据
    imgs = imgs.cuda()
    given_masks = [sta_mask.cuda()] + [None] * (L - 1)
    dist = get_dist(H, W)

    print("所有数据已处理完毕，即将进行预测")
    vos_out = model_predict(model_name, imgs, given_masks, dist.unsqueeze(0), sta_id)
    # print(vos_out["masks"].shape)
    save_predict(vos_out, out_path, video_name)
    #
    #
    # # #处理《分割结果》-得到《心脏图像量化数据》#
    # print("正在进行视频分割分析")
    predict_list = predict_deal(out_path + "predicts/")
    segment_properties = segment_analysis(predict_list)  # 调用 image_quantify.py 里的segment_analysis函数 将分割结果进行分割分析，获取椭圆轴、面积、等效直径属性
    seg_area = [seg['area'] for seg in segment_properties]

    # #处理《心脏图像量化数据》-得到《心脏机理量化数据》
    print("正在进行心室结构分析")
    systole_list = get_systole_frames(seg_area)  # 调用heart_quantify.py里的get_systole_frames函数，获取收缩
    diastole_list = get_diastole_frames(seg_area)  # 调用heart_quantify.py里的get_diastole_frames函数，获取舒张
    ventricular_dimensions = get_ventricular_dimensions(fps, systole_list, diastole_list, segment_properties) # 调用heart_quantify.py里的get_ventricular_dimensions函数，返回最大最小收缩体积，心搏量，射血分数，心率等数据

    save_3D(segment_properties, out_path)  # 保存 心脏跳动3D建模图 # 产生3D_spheroids文件夹
    write_to_csv(segment_properties, out_path, csv_name='segment_properties')  # 保存 心脏分割数据列表#           产生segment_properties.csv
    write_to_csv(ventricular_dimensions, out_path, csv_name='ventricular_dimensions')  # 保存 心脏机理参数列表#           产生ventricular_dimensions.csv
    save_electrocardiogram(out_path, seg_area, fps)  # 保存 心电图 # 产生electrocardiogram文件夹
    save_visualization(out_path, fps)           # 保存 分割可视化图 # 产生predict_visual文件夹
    print("全部处理结束")


if __name__ == '__main__':
    main()

