import numpy as np
from matplotlib import pyplot as plt

def hypothesis(theta_0, theta_1, x):
    return theta_0 + theta_1*x

def cost(parameters, features, labels):
    total = 0
    n = len(features)
    for i in range(0, n):
        total += (((parameters[0] + parameters[1]*features[i])-labels[i])**2)
    return total/2*n;

theta_0 = np.arange(-1.2,1.4,0.1)
theta_1 = np.arange(0.1,1.2,0.1)
features = np.arange(1,6.5,0.5)
labels = np.array([0.16023453,0.27728321,0.36117187,0.4802539,
            0.44229119,0.59856803,0.79987169,0.82492895,
            0.79536605,0.87930235,0.90780985],float)

min_j = 9999
min_aux = 0
theta_0_min = 0
theta_1_min = 0

for i in range(0, len(theta_0)):
    for j in range(0, len(theta_1)):
        min_aux = cost(np.array([theta_0[i],theta_1[j]]),features,labels)
        if(min_aux < min_j):
            min_j = min_aux
            theta_0_min = theta_0[i]
            theta_1_min = theta_1[j]

y_aux = []

for i in features:
    y_aux.append(hypothesis(theta_0_min, theta_1_min, i))

y_estimator = np.array(y_aux)

print("theta_0:", theta_0_min,"thetha_1:",theta_1_min)

plt.plot(features,labels)
plt.plot(features,y_estimator)
plt.xlabel('Features Voltage(V)')
plt.ylabel('Labels Current(A)')
plt.title('Regresión Lineal')
plt.grid()
plt.show()
