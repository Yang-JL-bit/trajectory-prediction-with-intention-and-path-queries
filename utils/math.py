'''
Author: Yang Jialong
Date: 2024-11-20 09:50:24
LastEditTime: 2024-11-20 17:07:29
Description: 数学函数
'''

import numpy as np
import math
 
 
def gaussian_elimination(A, b):
    """
    高斯消元法解线性方程组 Ax = b
    参数：
    A: 系数矩阵，形状为 (n, n)
    b: 右侧常数向量，形状为 (n,)
    返回：
    x: 解向量，形状为 (n,)
    """
    n = len(b)
 
    # 将系数矩阵和右侧常数合并
    Ab = np.column_stack((A.astype(float), b.astype(float)))
 
    # 前向消元
    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
 
    # 回代求解
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i:n], x[i:n])) / Ab[i, i]
    return x


'''
description: 
param {*} coeffs
param {int} order
param {*} x
return {*}
'''
def derivative(coeffs, order: int, x):
    if order < 0:
        raise ValueError("阶数必须是非负整数")
    
    n = len(coeffs) - 1  # 多项式的最高次数
    if order > n:
        # 超过最高阶次的导数恒为0
        return 0

    # 计算 n 阶导数的系数
    derived_coeffs = coeffs[:]
    for _ in range(order):
        derived_coeffs = [c * (n - i) for i, c in enumerate(derived_coeffs[:-1])]
        n -= 1
    
    # 计算多项式值
    result = sum(c * (x ** (n - i)) for i, c in enumerate(derived_coeffs))
    return result
