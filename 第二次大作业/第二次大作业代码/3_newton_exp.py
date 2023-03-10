from sympy import *
import numpy as np
import math
from matplotlib import pyplot

# 保存函数表达式,偏导数表达式,Hessian矩阵表达式
X, Y = symbols('X, Y')
F = math.e ** (X + 3 * Y - 0.1) + math.e ** (X - 3 * Y - 0.1) + math.e ** (-X - 0.1)
Fx = diff(F, X)
Fy = diff(F, Y)
Fxx = diff(Fx, X)
Fxy = diff(Fx, Y)
Fyx = diff(Fy, X)
Fyy = diff(Fy, Y)
# 求出全局最优解
x_optimal = -log(2, math.e) / 2
y_optimal = 0
p_optimal = F.subs({X: x_optimal, Y: y_optimal})

# 获取牛顿下降方向
def getd(x, y):
    Hxx = Fxx.subs({X: x, Y: y})
    Hxy = Fxy.subs({X: x, Y: y})
    Hyy = Fyy.subs({X: x, Y: y})
    Hessian = np.array([[Hxx, Hxy],
                        [Hxy, Hyy]], dtype = 'float64')

    dx = Fx.subs({X: x, Y: y})
    dy = Fy.subs({X: x, Y: y})
    dF = np.array([[dx],
                   [dy]], dtype = 'float64')
    dnt = -np.matmul(np.linalg.inv(Hessian),dF)
    return dnt

# 计算牛顿迭代过程(初始点(x0,y0),误差上界=eps)
def cal(x0, y0, alpha, beta, eps):
    k = 0
    x, y = x0, y0
    K = np.array([0])
    Loss = np.array([F.subs({X: x, Y: y}) - p_optimal])

    while(1):
        dnt = getd(x, y)
        dx = dnt[0][0]
        dy = dnt[1][0]
        # 计算牛顿减少量
        Hxx = Fxx.subs({X: x, Y: y})
        Hxy = Fxy.subs({X: x, Y: y})
        Hyy = Fyy.subs({X: x, Y: y})
        Hessian = np.array([[Hxx, Hxy],
                            [Hxy, Hyy]], dtype='float64')
        lamda2 = np.matmul(np.transpose(dnt), np.matmul(Hessian, dnt))
        # 终止条件
        if (lamda2 / 2 <= eps):
            break
        # 回溯直线搜索
        t = 1
        fnew = F.subs({X: x + t * dx, Y: y + t * dy})
        fk = F.subs({X: x, Y: y})
        incre = -alpha * lamda2
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

alpha = 0.1
beta = 0.7
eps = 1e-6
# 计算九个初始点的误差下降路径
pyplot.subplot(3, 3, 1)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(-2, 2, alpha, beta, eps)
pyplot.title("initial: (-2,2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 2)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(0, 2, alpha, beta, eps)
pyplot.title("initial: (0,2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 3)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(2, 2, alpha, beta, eps)
pyplot.title("initial: (2,2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 4)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(-2, 0, alpha, beta, eps)
pyplot.title("initial: (-2,0) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 5)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(0, 0, alpha, beta, eps)
pyplot.title("initial: (0,0) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 6)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(2, 0, alpha, beta, eps)
pyplot.title("initial: (2,0) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 7)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(-2, -2, alpha, beta, eps)
pyplot.title("initial: (-2,-2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 8)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(0, -2, alpha, beta, eps)
pyplot.title("initial: (0,-2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)

pyplot.subplot(3, 3, 9)
pyplot.yscale('log')
pyplot.tight_layout()
k, K, Loss = cal(2, -2, alpha, beta, eps)
pyplot.title("initial: (2,-2) total iteration = " + str(k))
pyplot.grid()
pyplot.scatter(K, Loss, color = "r")
pyplot.plot(K, Loss)
pyplot.show()