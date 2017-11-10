import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

alpha = 0.95

def share_proba(alpha, repetitions):
    return ((repetitions + 5) ** 2) * (alpha ** (repetitions + 5))

mpl.rcParams['legend.fontsize'] = 10
#fig = plt.figure()
#ax = fig.gca(projection='3d')

x = []
y = []
n = 1000
m = 100


z = np.zeros([m+1,n+1])

for i in range(n+1):
    x.append(i /(1. * n))

for j in range(m+1):
    y.append(j)

for i in range(n + 1):
    for j in range(m + 1):
        z[j,i]=(i / (1. * n) * share_proba(alpha, j))
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z, )
ax.set_xlabel('Quality')
ax.set_ylabel('Repetitions')
ax.set_zlabel('Share probability')

#ax.scatter(xs=q, ys=f, zs=p, label = 'share proba')
plt.show()


