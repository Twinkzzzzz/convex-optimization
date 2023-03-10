import numpy as np
from matplotlib import pyplot
import math
import pickle
import os
# 初始化A,xhat,b,KKT matrix
A = np.random.rand(30,100)
while (np.linalg.matrix_rank(A) < 30):
    A = np.random.rand(30, 100)
xhat = np.random.rand(100)
b = np.matmul(A, xhat)
KKTmat = np.zeros([130, 130])
# 初始化KKt矩阵中不变的部分
for i in range(100):
    for j in range(100,130):
        KKTmat[i][j] = A[j - 100][i]
for i in range(100,130):
    for j in range(100):
        KKTmat[i][j] = A[i - 100][j]
# 原问题F在X处的取值
def getF(X):
    Q = 0
    for i in range(100):
        Q += X[i] * np.math.log(X[i], np.e)
    return Q
# 对偶函数G在V处的取值
def getG(V):
    Q = 0
    for i in range(100):
        Q += np.e**(-np.dot(np.transpose(A)[i], V))
    Q /= np.e
    Q += np.dot(V, b)
    return Q
# Lagrange函数在(X,V)处的取值
def getL(X, V):
    Q = 0
    for i in range(100):
        Q += X[i] * np.math.log(X[i], np.e)
    Q += np.dot(V, np.matmul(A, X) - b)
    return Q
# 更新在X处的KKT矩阵
def getKKTmat(X):
    for i in range(100):
        KKTmat[i][i] = 1 / X[i]
# 获取原函数F在X处的梯度
def getgradientF(X):
    Q = np.zeros(100)
    for i in range(100):
        Q[i] = 1 + np.math.log(X[i], np.e)
    return Q
# 获取对偶函数G在V处的梯度
def getgradientG(V):
    return b - np.matmul(A, np.exp(-np.matmul(np.transpose(A), V))) / np.e
# 获取Lagrange函数当V固定时在X处的梯度
def getgradientL(X, V):
    Q = np.ones(100)
    for i in range(100):
        Q[i] += np.math.log(X[i], np.e)
    Q += np.matmul(np.transpose(A), V)
    return Q
# 获取原函数F在X处的Hessian矩阵
def getHessianF(X):
    Hessian = np.diag(1 / X)
    return Hessian
# 获取对偶函数G在V处的Hessian矩阵
def getHessianG(V):
    return np.matmul(A, np.matmul(np.diag(np.exp(-np.matmul(np.transpose(A), V))), np.transpose(A)))
# 获取Lagrange函数L在V固定时,在X处的Hessian矩阵
def getHessianL(X, V):
    return np.diag(1 / X)
# 由于原问题对定义域有限制,判断X是否在定义域中
def judgedomain(X):
    flag = True
    for it in X:
        if (it <= 0):
            flag = False
            break
    return flag
# 可行初始点的下降过程计算,初始点为x0
def nt_feasible(x0, alpha, beta, eps):
    x = x0
    k = 0
    # 记录下函数值,回溯直线搜索的总搜索次数随迭代次数的变化
    K = np.array([0])
    Loss = np.array([getF(x)])
    Iteration = np.array([0])
    totsearch = 0
    zerovec = np.zeros(30)
    while (1):
        #获取KKt矩阵,梯度,求出牛顿下降方向和牛顿减少量
        getKKTmat(x)
        KKTinv = np.linalg.inv(KKTmat)
        gre = np.zeros(100) - getgradientF(x)
        dnt = np.matmul(KKTinv, np.concatenate((gre, zerovec)))[0:100]
        Hessian = getHessianF(x)
        lamda2 = np.matmul(np.transpose(dnt), np.matmul(Hessian, dnt))
        # 判断终止条件
        if (lamda2 / 2 <= eps):
            break
        # 回溯直线搜索
        t = 1
        fnew = getF(x + t * dnt)
        fk = getF(x)
        incre = -alpha * lamda2
        while (fnew > fk + t * incre):
            t *= beta
            totsearch += 1
            fnew = getF(x + t * dnt)
        # 迭代过程
        x += t * dnt
        k = k + 1
        K = np.append(K, k)
        Iteration = np.append(Iteration, totsearch)
        Loss = np.append(Loss, fnew)
    # 输出迭代次数和结果,用于对比验证
    print("-----nt_feasible_initial-----")
    print("total iteration:", k)
    print("final p*:", fnew)
    return k, K, Loss, Iteration
# 不可行初始点的迭代过程计算,初始点为(x0,y0),均为随机生成
def nt_infeasible(x0, v0, alpha, beta, eps):
    x = x0
    v = v0
    k = 0
    K = np.array([0])
    Iteration = np.array([0])
    Loss = np.array([getF(x)])
    vec1 = np.zeros(30)
    totsearch = 0
    while (1):
        # 获取KKT矩阵
        getKKTmat(x)
        KKTinv = np.linalg.inv(KKTmat)
        # (gre,vec1)组成残差r
        gre = np.zeros(100) - getgradientF(x) - np.matmul(np.transpose(A), v)
        vec1 = b - np.matmul(A, x)
        r = np.concatenate((gre, vec1))
        # 获取x和v的牛顿下降方向
        dnt = np.matmul(KKTinv, r)
        dntx = dnt[0:100]
        dntv = dnt[100:130]
        rnormk = np.linalg.norm(r)
        # 终止条件判断
        if (rnormk <= eps and np.linalg.norm(vec1) <= eps):
            break
        # 回溯直线搜索阶段
        # 当x + t * dnt不在定义域中时需要一直迭代搜索
        # 通过给新的残差范数赋一个比原残差范数更大的值实现
        t = 1
        if (not judgedomain(x + t * dntx)):
            rnormnew = rnormk + 1
        else:
            gre = np.zeros(100) - getgradientF(x + t * dntx) - np.matmul(np.transpose(A), v + t * dntv)
            vec1 = b - np.matmul(A, x + t * dntx)
            r = np.concatenate((gre, vec1))
            rnormnew = np.linalg.norm(r)
        while (rnormnew > rnormk * (1 - alpha * t)):
            t *= beta
            totsearch += 1
            if (not judgedomain(x + t * dntx)):
                rnormnew = rnormk + 1
            else:
                gre = np.zeros(100) - getgradientF(x + t * dntx) - np.matmul(np.transpose(A), v + t * dntv)
                vec1 = b - np.matmul(A, x + t * dntx)
                r = np.concatenate((gre, vec1))
                rnormnew = np.linalg.norm(r)
        # 迭代过程
        x += t * dntx
        v += t * dntv
        k = k + 1
        K = np.append(K, k)
        Iteration = np.append(Iteration, totsearch)
        Loss = np.append(Loss, getF(x))
    # 输出总迭代次数和最终结果,用于对比验证
    print("-----nt_infeasible_initial-----")
    print("total iteration:", k)
    print("final p*:", Loss[-1])
    return k, K, Loss, Iteration

def nt_dual(x0, v0, alpha, beta, eps):
    # 阶段1: 对v进行牛顿下降
    k = 0
    v = v0
    K = np.array([0])
    totsearch = 0
    Iteration = np.array([0])
    Loss = np.array([getG(v)])
    while (1):
        # 获取对偶函数的Hessian矩阵和梯度
        # 计算牛顿下降方向和牛顿减少量
        Hessian = getHessianG(v)
        gre = getgradientG(v)
        dnt = -np.matmul(np.linalg.inv(Hessian), gre)
        lamda2 = np.matmul(np.transpose(dnt), np.matmul(Hessian, dnt))
        # 判断终止条件
        if (lamda2 / 2 <= eps):
            break
        # 回溯直线搜索
        t = 1
        fk = getG(v)
        fnew = getG(v + t * dnt)
        incre = -alpha * lamda2
        while (fnew > fk + t * incre):
            t = t * beta
            totsearch += 1
            fnew = getG(v + t * dnt)
        #迭代过程
        v += t * dnt
        k = k + 1
        K = np.append(K, k)
        Iteration = np.append(Iteration, totsearch)
        Loss = np.append(Loss, fnew)
    # 阶段2: 固定v的值,对x进行牛顿下降
    k2 = 0
    x = x0
    K2 = np.array([0])
    totsearch2 = 0
    Iteration2 = np.array([0])
    Loss2 = np.array([getL(x, v)])
    while (1):
        # 获取Lagrange函数当v固定时的Hessian矩阵和梯度
        # 计算牛顿下降方向和牛顿减少量
        Hessian = getHessianL(x, v)
        gre = getgradientL(x, v)
        dnt = -np.matmul(np.linalg.inv(Hessian), gre)
        lamda2 = np.matmul(np.transpose(dnt), np.matmul(Hessian, dnt))
        # 判断终止条件
        if (lamda2 / 2 <= eps):
            break
        # 回溯直线搜索
        # 当x + t * dnt不在定义域中时需要一直迭代搜索
        # 通过给新的函数值赋一个比函数值更大的值实现
        t = 1
        fk = getL(x, v)
        incre = -alpha * lamda2
        if (not judgedomain(x + t * dnt)):
            fnew = fk + 1
        else:
            fnew = getL(x + t * dnt, v)
        while (fnew > fk + t * incre):
            t = t * beta
            totsearch2 += 1
            if (not judgedomain(x + t * dnt)):
                fnew = fk + 1
            else:
                fnew = getL(x + t * dnt, v)
        # 迭代过程
        x += t * dnt
        k2 = k2 + 1
        K2 = np.append(K2, k2)
        Iteration2 = np.append(Iteration2, totsearch2)
        Loss2 = np.append(Loss2, fnew)
    # 输出总迭代次数和最终结果，用于对比验证
    # 返回两次迭代的所有信息
    print("-----nt_dual-----")
    print("total iteration:", k + k2)
    print("final p*:", Loss2[-1])
    return k, K, Loss, Iteration, k2, K2, Loss2, Iteration2

eps = 1e-8
alpha = 0.8
beta = 0.7

# 可行初始点
x0 = xhat
k, K, Loss, Ite = nt_feasible(x0, alpha, beta, eps)
pyplot.subplot(2, 4, 1)
pyplot.grid()
pyplot.title("nt_feasible_initial")
pyplot.xlabel("k")
pyplot.ylabel("F")
pyplot.plot(K, Loss)
pyplot.scatter(K, Loss, color = "r")
pyplot.subplot(2, 4, 5)
pyplot.xlabel("k")
pyplot.ylabel("times to search tk")
pyplot.grid()
pyplot.plot(K, Ite)
pyplot.scatter(K, Ite, color = "r")

# 不可行初始点
x0 = np.random.rand(100)
v0 = np.random.rand(30)
k, K, Loss, Ite = nt_infeasible(x0, v0, alpha, beta, eps)
pyplot.subplot(2, 4, 2)
pyplot.title("nt_infeasible_initial")
pyplot.xlabel("k")
pyplot.ylabel("F")
pyplot.grid()
pyplot.plot(K, Loss)
pyplot.scatter(K, Loss, color = "r")
pyplot.subplot(2, 4, 6)
pyplot.xlabel("k")
pyplot.ylabel("times to search tk")
pyplot.grid()
pyplot.plot(K, Ite)
pyplot.scatter(K, Ite, color = "r")

# 对偶方法
x0 = np.random.rand(100)
v0 = np.random.rand(30)
k, K, Loss, Ite, k2, K2, Loss2, Ite2 = nt_dual(x0, v0, alpha, beta, eps)
pyplot.subplot(2, 4, 3)
pyplot.title("nt_dual_period1(v)")
pyplot.xlabel("k(v)")
pyplot.ylabel("F_dual(v)")
pyplot.grid()
pyplot.plot(K, Loss)
pyplot.scatter(K, Loss, color = "r")
pyplot.subplot(2, 4, 7)
pyplot.xlabel("k")
pyplot.ylabel("times to search tk")
pyplot.grid()
pyplot.plot(K, Ite)
pyplot.scatter(K, Ite, color = "r")

pyplot.subplot(2, 4, 4)
pyplot.title("nt_dual_period1(x)")
pyplot.xlabel("k(x)")
pyplot.ylabel("F_dual(x)")
pyplot.grid()
pyplot.plot(K2, Loss2)
pyplot.scatter(K2, Loss2, color = "r")
pyplot.subplot(2, 4, 8)
pyplot.xlabel("k")
pyplot.ylabel("times to search tk")
pyplot.grid()
pyplot.plot(K2, Ite2)
pyplot.scatter(K2, Ite2, color = "r")
pyplot.show()