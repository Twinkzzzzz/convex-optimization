from sympy import *
import numpy as np
import pickle
import os
import math
from matplotlib import pyplot

# 存储函数表达式和函数偏导数表达式
X, Y = symbols('X, Y')
F = math.e ** (X + 3 * Y - 0.1) + math.e ** (X - 3 * Y - 0.1) + math.e ** (-X - 0.1)
Fx = diff(F, X)
Fy = diff(F, Y)
# 全局最优值
x_optimal = -log(2, math.e) / 2
y_optimal = 0
p_optimal = F.subs({X: x_optimal, Y: y_optimal})
# 获取下降方向 (负梯度方向)
def getd(x, y):
    dx = Fx.subs({X: x, Y: y})
    dy = Fy.subs({X: x, Y: y})
    return -dx, -dy
# 计算迭代过程 (初始点(x0,y0),误差eps)
def cal(x0, y0, alpha, beta, eps):
    # 初始点
    k = 0
    x, y = x0, y0
    K = np.array([0])
    Loss = np.array([F.subs({X: x, Y: y}) - p_optimal])
    while(1):
        # 判断终止条件
        dx, dy = getd(x, y)
        if (dx * dx + dy * dy <= eps * eps):
            break
        # 回溯直线搜索
        t = 1
        fnew = F.subs({X: x + t * dx, Y: y + t * dy})
        fk = F.subs({X: x, Y: y})
        incre = -alpha * (dx * dx + dy * dy)
        while (fnew > fk + t * incre):
            t *= beta
            fnew = F.subs({X: x + t * dx, Y: y + t * dy})
        # 迭代过程
        x += t * dx
        y += t * dy
        k += 1
        # 记录误差下降路径
        K = np.append(K, k)
        Loss = np.append(Loss, fnew - p_optimal)
    return k, K, Loss

# 画迭代次数热点图
def draw1(x0, y0, t ,eps):
    # 计算alpha从0.1~0.4,beta从0.1~0.9的迭代次数
    if (os.path.isfile("./resultk_" + str(x0) + "_" + str(y0) + "_" + str(eps) + ".pkl")):
        file = open("./resultk_" + str(x0) + "_" + str(y0) + "_" + str(eps) + ".pkl", "rb")
        resultk = pickle.load(file)
        file.close()
    else:
        resultk = np.zeros([30, 80])
        for i in range(0, 30):
            for j in range(0, 80):
                resultk[i][j], tp, tq = cal(x0, y0, i * 0.01 + 0.1, j * 0.01 + 0.1, eps)

        file = open("./resultk_" + str(x0) + "_" + str(y0) + "_" + str(eps) + ".pkl", "wb")
        pickle.dump(resultk, file)
        file.close()
    # 画迭代次数热点图
    Alpha = np.array([0.1, 0.2, 0.3, 0.4])
    Beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pyplot.subplot(3, 3, t)
    pyplot.title("initial:(" + str(x0) + "," + str(y0) + ")")
    pyplot.xlabel("beta")
    pyplot.ylabel("alpha")
    pyplot.yticks(np.array([0, 9, 19, 29]), Alpha)
    pyplot.xticks(np.array([0, 9, 19, 29, 39, 49, 59, 69, 79]), Beta)
    pyplot.imshow(resultk, origin = "lower")
    pyplot.colorbar()

# 画误差下降路径图
def draw2(x0, y0, alpha, beta, t ,eps):
    k, K, Loss = cal(x0, y0, alpha, beta, eps)
    pyplot.subplot(3, 3, t)
    pyplot.tight_layout()
    pyplot.yscale('log')
    pyplot.xlabel("k")
    pyplot.ylabel("Loss")
    pyplot.plot(K, Loss)
    pyplot.scatter(K, Loss, color = "r")
    pyplot.title("alpha=" + str(alpha) + "  beta=" + str(beta) + "  total iteration=" + str(k))

eps = 0.000001
# 从不同的初始点开始迭代,考察alpha和beta对迭代次数的影响
ra = 2
draw1(-ra, ra, 1, eps)
draw1(0, ra, 2, eps)
draw1(ra, ra, 3, eps)
draw1(-ra, 0, 4, eps)
draw1(0, 0, 5, eps)
draw1(ra, 0, 6, eps)
draw1(-ra, -ra, 7, eps)
draw1(0, -ra, 8, eps)
draw1(ra, -ra, 9, eps)
pyplot.show()
# 从同一初始点开始迭代,考察alpha和beta对误差下降情况的影响
x0, y0 = 4, 0
draw2(x0, y0, 0.1, 0.3, 1, eps)
draw2(x0, y0, 0.1, 0.5, 2, eps)
draw2(x0, y0, 0.1, 0.7, 3, eps)
draw2(x0, y0, 0.2, 0.3, 4, eps)
draw2(x0, y0, 0.2, 0.5, 5, eps)
draw2(x0, y0, 0.2, 0.7, 6, eps)
draw2(x0, y0, 0.3, 0.3, 7, eps)
draw2(x0, y0, 0.3, 0.5, 8, eps)
draw2(x0, y0, 0.3, 0.7, 9, eps)
pyplot.show()