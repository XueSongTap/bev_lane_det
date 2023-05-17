import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
'''
这段代码主要实现了一些用于处理三维点云数据的函数。

第一行导入了json模块，用于处理json格式的数据。

第二行导入了numpy模块，并将其命名为np，用于处理多维数组和矩阵。

第三行导入了torch模块，用于深度学习相关的计算。

第四行导入了matplotlib中的pyplot模块，并将其命名为plt，用于绘制图形。

第五行导入了scipy中的CubicSpline模块，用于进行三次样条插值。

接下来定义了一个名为mean_col_by_row_with_offset_z的函数，该函数接受三个参数：seg、offset_y和z。其中，seg是一个二维的numpy数组，表示分割后的点云数据；offset_y也是一个二维的numpy数组，表示y轴的偏移量；z是一个二维的numpy数组，表示z轴的坐标。该函数首先判断seg的维度是否为2，然后提取出所有的center_ids（即seg中所有不为0的id），并遍历每个center_id。对于每个center_id，遍历seg中的每一行，并找到该行中所有等于center_id的位置，将其保存在x_op中。然后计算每个位置对应的y值和z值，并根据偏移量计算出每个位置的x值。最后，将所有的x、y和z值保存在一个列表中，并返回该列表。

接下来定义了一个名为bev_instance2points_with_offset_z的函数，该函数接受四个参数：ids、max_x、meter_per_pixal和offset_y。其中，ids是一个numpy数组，表示点云数据；max_x表示x轴的最大值；meter_per_pixal是一个元组，表示每个像素对应的实际距离；offset_y是一个二维的numpy数组，表示y轴的偏移量。该函数首先计算出图像中心的位置，然后调用mean_col_by_row_with_offset_z函数，将其返回值保存在lines中。接着，遍历lines中的每个元素，将其x、y和z值分别保存在x、y和z中，并进行一系列的计算，最终得到点云数据的x、y和z坐标，并进行三次样条插值。最后，将所有的x、y、z和插值函数保存在一个列表中，并返回该列表。
'''

def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []
        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # mean pos
                cols.append(x_op)
                rows.append(y_op + 0.5)
        lines.append((cols, rows, z_val))
    return lines



def bev_instance2points_with_offset_z(ids: np.ndarray, max_x=50, meter_per_pixal=(0.2, 0.2), offset_y=None, Z=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)
    points = []
    # for i in range(1, ids.max()):
    for y, x, z in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x
        y = y * meter_per_pixal[1]
        y -= center * meter_per_pixal[1]
        x = x * meter_per_pixal[0]

        y *= -1.0  # Vector is from right to left
        if len(x) < 2:
            continue
        spline = CubicSpline(x, y, extrapolate=False)
        points.append((x, y, z, spline))
    return points


