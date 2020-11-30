import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

N = 28

x = np.linspace(-5, 5-10/N, N)

print(10/N)
print(x)

[X,Y] = np.meshgrid(x,x)

#print(X)
#print(Y)

plt.plot(X, Y,
         color='red',  # 全部点设置为红色
         marker='.',  # 点的形状为圆点
         linestyle='')  # 线型为空，也即点与点之间不用线连接
#plt.grid(True)
plt.show()

#关于meshgrid的解释，可以参考：https://blog.csdn.net/lllxxq141592654/article/details/81532855

thet = np.arctan2(Y, X)

print(thet)

fig = plt.figure()
ax = Axes3D(fig)

plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(X, Y, thet, rstride=1, cstride=1, cmap='rainbow')
plt.show()

#二元函数的绘制，可以参考：https://blog.csdn.net/your_answer/article/details/79135076
# 及https://blog.csdn.net/SHU15121856/article/details/72590620