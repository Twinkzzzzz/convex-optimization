运行环境：
    python 3.9
    matplotlib
    numpy

1_gradient.py:
    可直接运行

2_gradient_exp.py:
    可调节两对比实验的起始点和epsilon
    对比实验一的起始点为(0,0)和以(0,0)为几何中心的正方形的四个顶点以及四条边的中点，正方形边长为2ra
    line 91: epsilon
    line 93: ra 调节对比实验一的起始点
    line 105: x0,y0 调节对比实验二的起始点

3_newton_exp.py:
    line 75: epsilon
    从 line 76 开始计算九种起始点的迭代误差下降路径，起始点坐标在cal()函数中设定