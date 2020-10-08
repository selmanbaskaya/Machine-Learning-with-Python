# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:43:17 2020

@author: Selman
"""

""" Libraries """
#Installing Libraries
import pandas as pd
import matplotlib.pyplot as plt
""" Libraries End"""

""" Code Section """
#Data Upload
data = pd.read_csv('datasets/salaries.csv')
print(data)

#DataFrame Slicing, ünvanları ve maaşları ayrı değişkenlere alıyoruz.
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]

#NumPy transform
X = x.values
Y = y.values

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, Y)

#x'e denk gelen y değerlerini grafik üzerinde gösteriyoruz, anlaşılabilirliği artırmak için renk veriyoruz.
plt.scatter(X, Y, color = 'brown')
#tahmin doğrumuzu grafik üzerinde çizdiriyoruz
plt.plot(x, lr.predict(X), color = 'blue')
plt.show()

'''
Polynomial Regression - degree = 2
PolynomialFeatures ile herhangi bir sayıyı polinomal olarak ifade edebiliriz.
'''
from sklearn.preprocessing import PolynomialFeatures
#2.dereceden bir polinom oluşturuyoruz.
pr = PolynomialFeatures(degree = 2)

x_p = pr.fit_transform(X)
print(x_p)

'''
2.dereceye dönüşüm yaotık eğittik ve yeni grafiğimizi çizdiriyoruz, ilk grafiğimize farkla çok daha iyi bir sonuç
almış olduk, aynı işlemleri dereceyi 4 yaparak denediğimizde ise neredeyse birebir aynı noktalardan geçtiğini 
görebiliriz. Bu tabii ki her model için aynı sonucu vermez, problemleriniz de bunları deneyerek farkları görebilirsiniz
'''
lr_2 = LinearRegression()
lr_2.fit(x_p, y)
plt.scatter(X, Y, color = 'brown')
plt.plot(x, lr_2.predict(pr.fit_transform(X)), color = 'blue')
plt.show()

#Polynomial Regression - degree = 4
from sklearn.preprocessing import PolynomialFeatures
pr_2 = PolynomialFeatures(degree = 4)

x_p_2 = pr_2.fit_transform(X)
print(x_p_2)

lr_3 = LinearRegression()
lr_3.fit(x_p_2, y)
plt.scatter(X, Y, color = 'brown')
plt.plot(x, lr_3.predict(pr_2.fit_transform(X)), color = 'blue')
plt.show()
