#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:52:43 2021

@author: mariod
"""
## Importando las librerí­as
import numpy as np
from matplotlib import pyplot as plt

def hipotesis(X,tetha):
    """
    X debe ser una matriz donde cada fila es una muestra de características
    tetha debe ser un vector columna con los coeficientes
    la salida de la función es un vector columna con las probabilidades condicionales para cada una de las muestras
    """
    z = X @ tetha
    hipo = 1/(1+np.exp(-z))
    return hipo

# x_indice es un vector con x1,x2,x3 en la pos i
# y_indice en un numero
def costo(y,X,tetha):
    """
    X debe ser una matriz donde cada fila es una muestra de características
    tetha debe ser un vector columna con los coeficientes
    y es un vector columna con los resultados a cada uno de los experimentos
    """
    costo = np.zeros_like(y,np.float64)
    vector_1 = np.asarray(y == 0)[:,0]
    vector_2 = np.asarray(y == 1 )[:,0]
    h_vec1 = hipotesis(X[vector_1,:],tetha)
    h_vec2 = hipotesis(X[vector_2,:],tetha)
    costo[vector_1] = -np.log(1-h_vec1)
    costo[vector_2] = -np.log(h_vec2) 
    return sum(costo)/y.shape[0]

def gradiente(y,X,tetha):
    """
    X debe ser una matriz donde cada fila es una muestra de características
    tetha debe ser un vector columna con los coeficientes
    y es un vector columna con los resultados a cada uno de los experimentos
    Retorna un vector columna con el gradiente para cada parámetro
    """
    vec_h_y = hipotesis(X,tetha)-y # vector columna
    grad = (1/y.shape[0])*(np.transpose(X)@vec_h_y) # vector columna
    return grad

def calc_param(X,y,theta_inicial,alpha,error_min,max_iteracion):
    parametros = theta_inicial
    graf_J = list()
    for i in range(max_iteracion):
        parametros = parametros - alpha*gradiente(y,X,parametros)
        J = costo(y,X,parametros)
        graf_J.append(J[0,0])
        print("costo",J[0,0])
        if(J<error_min):
            break
    return parametros, graf_J


x1 = np.asmatrix([-1,0,1,-2,0,2]).T
x2= np.asmatrix([1,0,1,1,-1,1]).T

X = np.concatenate([np.ones_like(x1),x1,x2,np.power(x1,2)],1)
#x = np.transpose(x)
y = np.asmatrix([1,1,1,0,0,0]).T
theta_inicial = np.asmatrix([1,1,-3,1]).T
alpha = 0.01*10
error_min = 0.01
max_iteracion = 100000

tetha,graf_J = calc_param(X,y,theta_inicial,alpha,error_min,max_iteracion)
##tetha= np.array([-2,0.06,1.2,-2.1])
##print(costo(y,x,tetha) )

plt.plot(list(range(len(graf_J))),np.log(np.asarray(graf_J)))
plt.show()

plt.scatter(np.asarray(X[:,1][0:3]),np.asarray(X[:,2][0:3]),color='b')
plt.scatter(np.asarray(X[:,1][3:6]),np.asarray(X[:,2][3:6]),color='r')
### Graficando y = f(x)
x = np.arange(-2.5,2.5,0.05)
y = (tetha[0,0]+tetha[1,0]*x+tetha[3,0]*np.power(x,2))/(-tetha[2,0])
plt.plot(x,y)

plt.grid(True)
plt.show()

print("Los parámetros son:",tetha)