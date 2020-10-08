"""
@author: Selman
"""

""" Libraries """
#Installing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
""" Libraries End"""

""" Code Section """
#Data Upload
veriler = pd.read_csv('datasets/salaries.csv')

#DataFrame Slicing
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#predicts

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

'''
Support Vector'lerin aykırı verilere karşı bağımlılığı yok dolayısıyla modelimizi daha doğru eğitebilmek adına
data scaling işlemleri yapmamız gerekiyor.
'''
#data scaling
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

'''
SVR
svr'ın kernel parametresi için rbf kullanacağız, polynomial linear yapabilirsiniz, bunları daha ayrıntılı öğrenmek
için sklearn dökümantasyonlarını kesinlikle incelemenizi öneririm.
fit fonksiyonunu ölçeklendirdiğimiz veriler üzerinden yapıyoruz
'''
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

#Öğrenme işlemleri biten modelimizi yeni test değerleri vererek tekrardan test ediyoruz.
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

'''
Decision Tree
dt'nin random_state parametresi için 0 kullanacağız, random-state'i linear reg. konusunda anlatmıştık.
'''
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

'''
Random Forest
rf'nin random_state parametresi için 0 kullanacağız, random-state'i linear reg. konusunda anlatmıştık.
n_estimators parametresi ise kaç tane decision tree çizeceğini söylemiş oluyoruz. (random forest dec. tree algoritmaları
                                                                                   ile çalışıyor).
'''
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()

