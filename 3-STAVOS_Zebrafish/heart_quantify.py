# -*- coding:utf-8 -*-
import numpy as np
from scipy.signal import find_peaks


def get_systole_frames(timeseries):
    """
    :param timeseries:随时间推移具有分段属性的数组
    :return:每个时间步的估计频率
    """
    peak_list = find_peaks(np.negative(timeseries), distance=6)
    # print('systole', peak_list)
    return peak_list


def get_diastole_frames(timeseries):
    """
    :param timeseries:随时间推移具有分段属性的数组
    :return:每个时间步的估计频率
    """

    peak_list = find_peaks(timeseries, distance=6)
    # print('diastole', peak_list)
    return peak_list


# scipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
# x: 带有峰值的信号序列
# height: 低于指定height的信号都不考虑
# threshold: 其与相邻样本的垂直距离
# distance: 相邻峰之间的最小水平距离, 先移除较小的峰，直到所有剩余峰的条件都满足为止。
# prominence: 个人理解是突起程度，详见peak_prominences
# width: 波峰的宽度，详见peak_widths
# plateau_size: 保证峰对应的平顶数目大于给定值
# peaks: x对应的峰值的索引

# properties:
# height--> ‘peak_heights’
# threshold-->‘left_thresholds’, ‘right_thresholds’
# prominence-->‘prominences’, ‘right_bases’, ‘left_bases’
# width-->‘width_heights’, ‘left_ips’, ‘right_ips’
# plateau_size-->‘plateau_sizes’, left_edges’, ‘right_edges’


def get_V(minorAxis, majorAxis):
    """
    ：参数
        minorAxis:带有短椭圆轴值的标量
        majorAxis:具有椭圆长轴值的标量
        camera_perspective:定义相机的视点,从而调用体积近似
    ：返回：
        心脏体积近似为px^3中的圆柱体
    """
    # 这个体积模型是第一近似，可以做得更复杂，
    # 一旦完全理解了相机的视角
    V = 1/6 * np.pi * majorAxis * np.power(minorAxis, 2)
    return V

def get_HR(fps, sys_0, dia_0, sys_1, dia_1):
    """
    ：参数
        fps:视频的每秒帧数
        sys_frame:找到收缩峰的帧
        dia_frame:找到舒张峰的帧
    :return:
        每个时间步的估计频率
    """
    sys_heart_rate = fps/(1+sys_1-sys_0)
    dia_heart_rate = fps/(1+dia_1-dia_0)
    HR = np.mean([sys_heart_rate, dia_heart_rate])*60
    return HR

def get_SV(sysV, diaV):
    """
    ：参数
        sysV:收缩容积(px^3)
        diaV:舒张容积(px^3)
    ：返回：
        笔划体积(px^3)
    """

    SV = diaV - sysV
    return SV

def get_FS(sys, dia):
    """
    :参数
        sys: 心脏收缩值的标量
        dia: 心脏舒张的的标量
    ：返回：
        左室短轴缩短率（%)fractional shortening
    """
    FS = 100 * abs(dia-sys) / dia
    return FS


def get_EF(sysV, diaV):
    """
    ：参数
        sysV:收缩容积(px^3)
        diaV:舒张容积(px^3)
    ：返回：
        心容量变化（%)
    """
    EF = 100 * (diaV-sysV) / diaV
    return EF








def get_ventricular_dimensions(fps, sys, dia, segment_properties):
    """
    ：参数
        fps:捕获视频的每秒帧数
        sys:具有收缩帧的数组
        dia:具有舒张框架的阵列
        df:已确定并保存段规格
        cameraperspective:str,相机的透视,为近似导出了不同的数学模型
    ：返回：
        .csv文件,所有心室尺寸均保存并显示在图像中
    """
    ventricular_dimension_list = []

    ellipse_minor_axis = [seg['ellipse minor axis'] for seg in segment_properties]
    ellipse_major_axis = [seg['ellipse major axis'] for seg in segment_properties]

    for frame in range(0, len(sys[0])-1):
        # print(frame)
        systolic_frame = sys[0][frame]
        diastolic_frame = dia[0][frame]
        minorFS = get_FS(ellipse_minor_axis[sys[0][frame]], ellipse_minor_axis[dia[0][frame]])
        majorFS = get_FS(ellipse_major_axis[sys[0][frame]], ellipse_major_axis[dia[0][frame]])
        sysV = get_V(ellipse_minor_axis[sys[0][frame]], ellipse_major_axis[sys[0][frame]])
        diaV = get_V(ellipse_minor_axis[dia[0][frame]], ellipse_major_axis[dia[0][frame]])
        EF = get_EF(sysV, diaV)
        SV = get_SV(sysV, diaV)
        HR = get_HR(fps, sys[0][frame], dia[0][frame], sys[0][frame + 1], dia[0][frame + 1])

        ventricular_dimension = {
                                        'sys_frame': systolic_frame,  # 收缩帧
                                        'dia_frame': diastolic_frame, # 舒张帧
                                        'minorFS [%]': minorFS, # minor fractional shortening
                                        'majorFS [%]': majorFS, # major fractional shortening
                                        'sysV [px^3]': sysV,    # 收缩体积
                                        'diaV [px^3]': diaV,    # 舒张体积
                                        'HR(heart rate)[bpm]': HR,  # 心率
                                        'SV(stroke volume)[px^3]': SV,          # 心搏量
                                        'EF(ejection fraction)[%]': EF,         # 射血分数

                                }

        ventricular_dimension_list.append(ventricular_dimension)

    return ventricular_dimension_list
