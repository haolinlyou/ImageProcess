# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : hog.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/13 13:47
# Copyright 2020 lk <lk123400@163.com>

import numpy as np
import cv2

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)


sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)


def conv(input_map, conv_kernal, stride=1, padding='SAME'):
    delta = conv_kernal.shape[0] // 2
    h, w = input_map.shape
    res = np.zeros((h, w), dtype=np.float32)
    if padding == 'SAME':
        input_map = pad(input_map, delta)
    for i in range(1, h + 1 - 2 * delta):
        for j in range(1, w + 1 - 2 * delta):
            res[i, j] = np.sum(input_map[i-1:i+1+delta, j-1:j+1+delta]*conv_kernal)
    return res


def sobel(gray_img, orited=0):
    if gray_img is None:
        return
    h, w = gray_img.shape
    gray_img = np.reshape(gray_img, (h,w)).astype(np.float32)
    if orited == 0:
        return conv(gray_img, sobel_x, 1)
    elif orited == 1:
        return conv(gray_img, sobel_y, 1)



def pad(gray_img, padding_size=1):
    return np.pad(gray_img, (padding_size, padding_size), 'constant')



def imread(path):
    return cv2.imread(path, 0)

def gamma_adjust(gray_img):
    pass

def get_cell():
    pass

def get_bins(theta, size):
    """
    将180度平均分成size份，相应的theta进入各类桶中
    :param theta:
    :param size:
    :return:
    """
    bins = np.zeros_like(theta, dtype=np.int)
    delta = np.pi/size
    for i in range(size):
        bins[np.where((theta >= delta * i) & (theta <= delta * (i + 1)))] = i
    return bins


def cal_gradient(gray_img):
    """
    使用sobel算子计算图像水平和垂直方向梯度
    """
    sobel_x_img = sobel(gray_img, 0)
    sobel_y_img = sobel(gray_img, 1)
    #将值为0的只为epsonal,防止计算梯度方向时出现除0操作
    sobel_x_img[sobel_x_img == 0] = 0.0001
    return sobel_x_img, sobel_y_img

def get_mag_theta(gradient_x, gradient_y):
    """
    计算梯度方向和幅值  theta = arctan(y/x)， 单位为弧度
                        mag = x**2 + y**2
    """
    theta = np.arctan(gradient_y / gradient_x)
    mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return theta, mag

def cal_gradient_histom(bins, mag, img_size, cell_size, bin_size, stride):
    """
    计算梯度直方图
    :param bins:         按梯度方向分区间后的值
    :param mag:          梯度幅值
    :param img_size:     图像的大小
    :param cell_size:    每个cell的大小
    :param bin_size:     bin的分类数量
    :return:
    """
    h, w = img_size
    hh = h//cell_size
    ww = w//cell_size
    hist = np.zeros((hh, ww, bin_size), dtype=np.float32)
    for y in range(hh):
        for x in range(ww):
            for j in range(cell_size):
                for i in range(cell_size):
                    hist[y, x, bins[y*stride+j, x*stride+i]] += mag[y*stride+j, x*stride+i]
    return hist


def histom_normalize(input_histom, img_size, cell_size, block_size, eps=1):
    """
    直方图归一化
    :param input_histom:
    :param img_size:
    :param cell_size:
    :param block_size:
    :param eps:
    :return:
    """
    delta = block_size//2
    h, w = img_size
    hh = h // cell_size
    ww = w // cell_size
    for y in range(hh):
        for x in range(ww):
            #print(np.sqrt(np.sum(input_histom[max(y-delta, 0):min(y+1+delta, hh),
            #                                   max(x-delta, 0):min(x+1+delta, ww)]**2)+eps))
            input_histom[y,x] /= np.sqrt(np.sum(input_histom[max(y-delta, 0):min(y+1+delta, hh),
                                                max(x-delta, 0):min(x+1+delta, ww)]**2)+eps)
    return input_histom





def hog(gray_img):
    """
    hog 特征实现
    :param path:
    :return:
    """
    #划分区间数
    bin_size = 9
    img_size = gray_img.shape
    #计算图像梯度，默认使用sobel算子
    sobel_x_img, sobel_y_img = cal_gradient(gray_img)
    #计算梯度幅值与角度，单位为弧度
    theta, mag = get_mag_theta(sobel_x_img, sobel_y_img)
    #无向
    theta[theta<0] += np.pi
    #将梯度方向分成不同的区间，bin_size表示将180度分成bin_size份
    bins = get_bins(theta, bin_size)
    #计算cell里梯度方向直方图，每个cell_size为8
    hist = cal_gradient_histom(bins, mag, img_size, 8, bin_size, 4)
    #梯度直方图归一化
    hist_normalize = histom_normalize(hist, img_size, 8, 3)

def test(gray_img):
    sobel_x_img = cv2.Sobel(gray_img,cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_img = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imshow("cv_sobel_x", sobel_x_img)
    cv2.imshow("cv_sobel_y", sobel_y_img)
    return sobel_x_img, sobel_y_img


if __name__ == '__main__':
    gray_img = imread('../data/lena.jpg')
    print(gray_img.shape)
    #cv_sobel_x_img, cv_sobel_y_img = test(gray_img)
    hog(gray_img)
    cv2.waitKey(0)





