import numpy as np
from matplotlib import pyplot

def cal(GM, eps, if_draw):
    print("gama =", GM, "if_draw =", if_draw)
    if (if_draw):
        # 迭代坐标路径图
        pyplot.subplot(2, 1, 1)
        pyplot.grid()
        # 画出坐标平面上函数值分布
        tmp = pow(((GM - 1) / (GM + 1)), 0)
        x1, x2 = tmp * GM, tmp * pow(-1, 0)
        pyplot.xlim(-x1 - 1, x1 + 1)
        pyplot.ylim(-x2 - 1, x2 + 1)
        p = np.arange(-x1 - 1.1, x1 + 1.1, 0.01)
        q = np.arange(-x2 - 1.1, x2 + 1.1, 0.01)
        P, Q = np.meshgrid(p, q)
        Z = 1 / 2 * (P * P + GM * Q * Q)
        ctf = pyplot.contourf(P, Q, Z, 50)
        pyplot.colorbar(ctf)

    X1 = np.empty(0)
    X2 = np.empty(0)
    F = np.empty(0)
    k = 0
    while (1):
        # 计算xk
        tmp = pow(((GM - 1) /  (GM + 1)), k)
        x1, x2 = tmp * GM, tmp * pow(-1, k)
        F = np.append(F, 1 / 2 * (pow(x1, 2) + GM *pow(x2, 2)))
        if (if_draw):
            pyplot.scatter(x1, x2, color = "r")
        # 记录坐标路径
        X1 = np.append(X1, x1)
        X2 = np.append(X2, x2)
        gre = np.array([x1, GM * x2])
        # 终止条件
        if (np.linalg.norm(gre) <= eps):
            break
        k += 1
    if (if_draw):
        # 画迭代坐标路径
        pyplot.plot(X1, X2)
        pyplot.title("GAMA=" + str(GM) + "  EPS=" + str(eps) + "  total iteration=" + str(k), fontsize = 20)
        # 误差下降路径图
        pyplot.subplot(2,1,2)
        #pyplot.yscale('log')
        pyplot.xlabel("k")
        pyplot.ylabel("Loss")
        pyplot.plot(np.arange(k + 1), F)
        # 选取个别结点记录误差值标签
        stride = int(k / 5)
        if(stride == 0):
            stride = 1
        for i in range(0, k + 1 - stride, stride):
            pyplot.text(i, F[i], str(i) + ": " + str(F[i]), rotation = 45)
            pyplot.scatter(i, F[i], color = "r")
        pyplot.text(k, F[k], str(k) + ": " + str(F[k]), rotation = 45)
        pyplot.scatter(k, F[k], color = "r")
        pyplot.show()
    return k

eps = 1e-6
# 计算个别gama值的迭代过程(True代表需要画图)
gama = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 50, 100]
ite = []
for gm in gama:
    k = cal(gm, eps, True)

# 计算从1/1000(左右)到1000(左右)的gama值的迭代次数
# 作出迭代次数与gama的关系,以2位底数是为了更多采样
gama = [1 / 1024, 1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8 , 1 / 4 , 1 / 2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ite = []
for gm in gama:
    k = cal(gm, eps, False)
    ite.append(k)

pyplot.xscale("log")
pyplot.plot(gama, ite)
pyplot.scatter(gama, ite, color = "r")
pyplot.xlabel("k")
pyplot.ylabel("iteration")
pyplot.title("eps = 1e-6")
for i in range(21):
    pyplot.text(gama[i], ite[i], str(gama[i]) + " , ite=" + str(ite[i]), rotation = 45)
pyplot.show()