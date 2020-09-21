import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 500)

Q = 225000
EI = 4500000
k = 40000000
L = np.power(4 * EI / k, 1 / 4)

y1 = -((Q * L ** 3) / (8 * EI)) * np.exp(-x/L)*(np.cos(abs(x)/L)+np.sin(np.abs(x)/L))
# y1 = np.exp(-x/L)*(np.cos(abs(x)/L)+np.sin(np.abs(x)/L))
m = Q * L / 4 * np.exp(-x/L)*(np.cos(abs(x)/L)-np.sin(np.abs(x)/L))


diff_x = np.diff(x)
# y1 = np.sin(x)
y2 = np.diff(y1) / diff_x
y3 = np.diff(y2) / diff_x[1:]

plt.plot(x, y1)
plt.plot(x[1:], y2)
plt.plot(x[2:], y3)
plt.scatter(x, m / EI)
plt.grid()
plt.show()

#
epsilon = 8e-5
E = 210e9
W = 330e-6
M = E * W * epsilon
print(M)
