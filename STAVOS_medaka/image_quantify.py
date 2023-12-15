# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np


# 输入图像，返回带阈值、选定和填充段的图像
def filtering(image):
    # 阈值化
    H, W = image.shape
    _, image = cv2.threshold(image, thresh=200, maxval=255, type=0)  # >thresh置为255白，    <=thresh置为0黑

    # 查找所有分割片段
    image = cv2.convertScaleAbs(image)  # 通过线性变换将数据转换成8位[uint8]

    # 打开
    kernel = np.ones((25, 25), np.uint8)
    # cv2.morphologyEx(src, op, kernel)# 进行各类形态学的变化,src传入的图片,op进行变化的方式,kernel表示方框的大小
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
    # 开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积

    # 输入图像，返回每条轮廓对应的属性及每条轮廓对应的属性
    segments, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # 计算最大面积分割片段
    max_segment = max(segments, key=cv2.contourArea, default=0)

    # 分割片段填充
    # 绘制轮廓,将图片全置为黑，然后将轮廓内部填充为白色
    image = cv2.drawContours(image=np.zeros((H, W)), contourIdx=-1, contours=max_segment, color=255, thickness=cv2.FILLED)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # 进行闭运算,先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。

    return image, max_segment  # 返回轮廓填充后的图片，最大面积分割片段



def segment_analysis(images):

    # 加载框架
    properties_list = []
    for img in images:
        # 分割预测（图像）的阈值化和滤波
        img, segment = filtering(img)
        # cv2.imwrite("img.png", img)
        # cv2.imwrite("seg.png", seg)
        # 分割片段面积
        if isinstance(segment, int):
            area = 1
            # fitting an ellipse returns ((x_center, y_center), (minor_axis, major_axis), angle)
            ellipse = ((1, 1), (1, 1), 1)
        else:
            area = cv2.contourArea(segment)
            # fitting an ellipse returns ((x_center, y_center), (minor_axis, major_axis), angle)
            ellipse = cv2.fitEllipse(segment)
        # 等效直径（等效直径是面积与轮廓面积相同的圆的直径。）
        equip_diameter = np.sqrt(4 * area / np.pi)
        # 将属性保存到字典（集合可能会增加）
        segment_properties = {'area': area, 'equivalentDiameter': equip_diameter, 'ellipse minor axis': ellipse[1][0], 'ellipse major axis': ellipse[1][1]}
        properties_list.append(segment_properties)
    return properties_list





def predict_deal(out_path):
    filenames = os.listdir(out_path)  # 获取当前路径下的文件名，返回List
    predict_list = []
    for i in range(len(filenames)):
        predict_file = out_path + "/" + filenames[i]
        img = cv2.imread(predict_file, 0)
        predict_list.append(img)
    return predict_list
