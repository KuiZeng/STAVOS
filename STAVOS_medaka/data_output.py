# -*- coding:utf-8 -*-
import os
import csv
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def create_all_dir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "frames/", exist_ok=True)
    os.makedirs(path + "predicts/", exist_ok=True)
    os.makedirs(path + "3D_spheroids/", exist_ok=True)
    os.makedirs(path + "predict_visual/", exist_ok=True)
    os.makedirs(path + "electrocardiogram/", exist_ok=True)
    return path


def write_to_csv(dict_data, path, csv_name):  # 写成csv格式
    # 将字典数组写入csv文件
    with open(path + csv_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(dict_data[0].keys())
        w.writerows(frame.values() for frame in dict_data)


def save_3D(segment_properties, out_path):
    """
    在3D中将估计的心脏绘制为扁长椭球形状
    将打印另存为.tif-image
    """
    print("正在保存心室3D模拟图")
    save_path = out_path + "3D_spheroids/"
    i = 0
    elevation = 40
    azimuth = -45
    for e in segment_properties:
        fig = plt.figure() # 新建画布！！！很重要
        ax = fig.add_subplot(111, projection='3d')

        # 定义长椭球
        theta, phi = np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15)
        THETA, PHI = np.meshgrid(theta, phi)

        a = e['ellipse minor axis'] / 2
        c = e['ellipse major axis'] / 2

        Z = a * np.sin(PHI) * np.cos(THETA)
        X = a * np.sin(PHI) * np.sin(THETA)
        Y = c * np.cos(PHI)

        # 绘制一个基本的线框图。
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Reds', edgecolor='none')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        plt.xticks(np.arange(-200, 200, 100))
        plt.yticks(np.arange(-200, 200, 100))

        ax.view_init(elevation, azimuth)
        # elevation += 1
        # azimuth -= 1

        fig.savefig(save_path + "/spheroid_{:05d}.png".format(i))
        i += 1
        plt.close()


    ## 生成gif
    filenames = os.listdir(save_path)
    with imageio.get_writer(out_path + '3D_spheroids.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.v2.imread(save_path + filename)
            writer.append_data(image)
    plt.clf()


def save_electrocardiogram(out_path, y, fps):
    print("正在保存心电图")
    plt.clf()
    save_path = out_path + "electrocardiogram/"


    n = len(y)
    x = range(n)
    plt.figure()
    plt.plot(x, y, color='r')
    plt.savefig(out_path + "electrocardiogram.png")
    mi, ma = min(y), max(y)

    ## 循环生成只有两个点一根线的图
    plt.clf()
    start = 0
    for i in range(n):
        plt.figure(figsize=(16, 12))
        plt.xlabel(f"frame (fps={fps})", fontsize=40)
        plt.ylabel("volume (px^3)", fontsize=40)

        plt.plot(x[start:i], y[start:i], linestyle='--', marker='o', color='r', markersize=12, linewidth=4)
        plt.xlim(start, start + 30)
        plt.ylim(mi-400, ma+400)
        plt.savefig(save_path + 'electrocardiogram_{:05d}.png'.format(i))
        plt.close()
        if i % 30 == 0:
            start = i

    ## 生成gif
    filenames = os.listdir(save_path)
    with imageio.get_writer(out_path + 'electrocardiogram.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.v2.imread(save_path + filename)
            writer.append_data(image)


def save_visualization(out_path, fps):
    print("正在保存分割可视化")
    img_path = out_path + "frames/"
    mask_path = out_path + "predicts/"
    save_path = out_path + "predict_visual/"

    img_names = os.listdir(img_path)  # 获取当前路径下的文件名，返回List
    mask_names = os.listdir(mask_path)  # 获取当前路径下的文件名，返回List


    img = cv2.imread(img_path + img_names[0])
    # mask = cv2.imread(mask_path + mask_names[0])
    size = (img.shape[1], img.shape[0])
    # video = cv2.VideoWriter(out_path + "predict_visual.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    for i in range(len(mask_names)):
        # print("正在绘制的照片:", img_names[i])
        img = cv2.imread(img_path + img_names[i])
        mask = cv2.imread(mask_path + mask_names[i])

        if img.shape != mask.shape:
            mask = cv2.resize(mask, size)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(mask, 1, 255, 0)
        contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
        seg_visual = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
        cv2.imwrite(save_path + "visual_{:05d}.png".format(i), seg_visual)
        # video.write(seg_visual)

    filenames = os.listdir(save_path)
    with imageio.get_writer(out_path + 'predict_visual.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.v2.imread(save_path + filename)
            writer.append_data(image)

