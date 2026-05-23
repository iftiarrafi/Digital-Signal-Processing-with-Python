# matplotlib
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-1, 1, 0.01)

#sine curve
y1 = np.sin(2 * np.pi * 2 * t)
plt.plot(t, y1)

#cosine curve
# y2 = np.cos(2 * np.pi * 2 * t)
# plt.plot(t, y2)

plt.xlabel("Time (t)")
#plt.ylabel("sin(2π * 2 * t)")
plt.title("Sine Curve")
plt.grid(True)
plt.show()
