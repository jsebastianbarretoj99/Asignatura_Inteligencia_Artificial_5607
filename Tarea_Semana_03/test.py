from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,6.5,0.5)

y = np.array([0.16023453,0.27728321,0.36117187,0.4802539,
            0.44229119,0.59856803,0.79987169,0.82492895,
            0.79536605,0.87930235,0.90780985],float)

plt.plot(x,y)
plt.xlabel('Features Voltage(V)')
plt.ylabel('Labels Current(A)')
plt.title('Regresión Lineal')
plt.grid()
plt.show()