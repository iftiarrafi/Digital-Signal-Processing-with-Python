#matplotlib

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,11,1);
print(x)
y1 = 2*x
y2 = 3*x
print(y1)
plt.plot(x,y1, color='blue' , linewidth='1')
plt.plot(x,y2, color='red' , linewidth='2')
plt.grid(True)
plt.xlabel("X-axis")



plt.show()