"""
@author: Selman
"""

""" Libraries """

"""
Installing Libraries
Her adımda bir önceki kod dosyasında açıkladığımız kısımları tekrar açıklamadan ilerleyeceğiz, sizleri de ezberden
uzak tutmak adına daha iyi bir yöntem olacağını düşünüyorum.
Ve başlıkları ingilizce anlamları ile yazarak biraz da genel kullanımlara aşina olmanızı istiyorum.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
""" Libraries End"""

""" Code Section """
#Data Preprocessing

#Data Upload
data = pd.read_csv('datasets/missing_values.csv')
print(data)

country = data.iloc[:,0:1].values
print(country) 

'''
sci - kit learn (sklearn)
Veri setimizi inceleyecek olursanız bazı satırlarda yaş değerinin boş olduğunu göreceksiniz, bu ise bizim karşılaşmak
istemediğimiz bir durumdur. sklearn'ün içerisinden SimpleImputer ile değeri olmayan alanlara atamalar yapacağız.
Bu gibi durumlarda birdem fazla seçenek kullanılabilir, en büyük, en küçük, ortalama veya 0 1 gibi rastgele 
değerler atanabilir, biz bu problemi çözmek için ortalama değerini kullanacağınız.

'''
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
age = data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)

'''
sci - kit learn (sklearn)
Kategorik veri tiplerinde dönüşüm
Country kolonunun altında üç farklı ülke ismi var fakat bizim veri setinde işlem yapabilmemiz için bunları sayısal
değerlere dönüştürmemiz gerekiyor.
Label Encoding işlemi her ülke için 1'den başlarak sayısal değer atamak için kullanılır. Fakat bu da istenilen bir
durum değil çünkü tr için bir us için 2 değerleri geldi, modelimiz öğrenmeye başlarken bu iki değeri kat farkı olarak
düşünebilir ve bu şekilde düşünerek modeli eğitmesi yanlış olacaktır. Bundan kurtulmak için ise onehotencoding yapacağız.
Label enc. sonucu -> 11...1 22...2 33...3 şeklinde oldu.
'''
from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
country[:,0] = label_encoding.fit_transform(data.iloc[:,0])
print(country)

'''
OneHotEncoding tr us fr verilerini etiket haline getirir ve her satır için 0 ve 1 değerlerini girer.
ohe ile yeni durum şu şekilde;
tr us fr
1  0  0
0  1  0
0  0  1
haline dönüştü.
'''
one_hot_encoding = preprocessing.OneHotEncoder()
country = one_hot_encoding.fit_transform(country).toarray()
print(country)

#country için yaptığımız label encoder ve onehotencoder işlemlerini gender kolonu için de yapacağız.
gen = data.iloc[:,-1:].values
print(gen) 

#Label Encoding işlemi her değer için 1'den başlarak sayısal değer atamak için kullanılır.
from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
gen[:,-1] = label_encoding.fit_transform(data.iloc[:,-1])
print(gen)

#Kolon başlıklarını etiketlere taşır, her etikete 1 veya 0 değeri verir.
one_hot_encoding = preprocessing.OneHotEncoder()
gen = one_hot_encoding.fit_transform(gen).toarray()
print(gen)


#Numpy dizilerinin DataFrame'e dönüştürülmesi işlemleri. Kolon başlığı ve index ekleyerek DataFrame oluşturuluyor.
result_counrty = pd.DataFrame(data = country, index = range(22), columns = ['fr', 'tr', 'us'])
print(result_counrty)

result_hwa = pd.DataFrame(data = age, index = range(22), columns = ['height', 'weight', 'age'])
print(result_hwa)

gender = data.iloc[:,-1].values
print(gender)

'''
veri setimize gender kolonunu eklerken tek bir sütunu almalıyız, hatırlarsanız yukarıda onehotencoding yaparken
erkek veya kadın olması durumu için iki ayrı kolon kullanmıştık, fakat birinin değerini bilmemiz diğer durumu
tahmin etmemize yaradığı için tek bir kolon kullanmamız yeterli olacaktır.
'''
result_gender = pd.DataFrame(data = gen[:, : 1], index = range(22), columns = ['gender'])
print(result_gender)

#DataFrame'leri birleştirme işlemi.
result = pd.concat([result_counrty, result_hwa], axis = 1)
print(result)

result2 = pd.concat([result, result_gender], axis = 1)
print(result2)

#Veriler train ve test için bölündü.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(result, result_gender, test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#boyu tahmin etmek istersek bu sefer bağımlı değişken olarak height kolonunu kullanacağız, bunu bir değişkene alalım
height = result2.iloc[:,3:4].values
print(height)

#daha sonra ise height kolonunun sağında ve solunda kalanları ayrı ayrı alıp bir sonraki adımda concat işlemi yapacağız.
left = result2.iloc[:,:3]
right = result2.iloc[:,4:]

data_concat = pd.concat([left, right], axis = 1)

#yemi bir veri seti oluşturmuş olduk aşağıdaki yapı ile de height kolonunu tahmin eden bir model tasarlıyor olacağız.
x_train, x_test, y_train, y_test = train_test_split(data_concat, height, test_size = 0.33, random_state = 0)

regressor_2 = LinearRegression()
regressor_2.fit(x_train, y_train)

y_pred_2 = regressor.predict(x_test)

'''
Geri Eleme
Her bir değişkenin sistem üzerine bir etkisi vardır. Bazı değişkenlerin sisteme etkisi yüksekken bazılarının azdır. 
Sisteme etkisi az olan bağımsız değişkenlerin ortadan kaldırılması daha iyi bir model kurmamıza olanak sağlar.
Backwar d Elimination yöntemini kullanarak daha iyi modeller oluşturabiliriz.

1-P değerini seçin ( Genellikle bu değer 0.05 olur)
2-Tüm bağımsız değişkenleri dahil ettiğiniz bir model kurun
3-Her bir bağımsız değişkenin p değeri incelenir. Eğer Pdeğeri model için belirlenenden daha büyük ise bu bağımsız değişken modelden çıkarılır. Tekrar çalıştırılır.
4-Bütün p değerleri belirlediğimiz değerden küçük ise modelimiz hazırdır.
'''
import statsmodels.api as sm

#veri setimizin en sonuna veri sayımızın satır sayısı ile aynı olacak yeni bir kolon ekliyoruz ve içerisine 1 yazıyoruz 
X = np.append(arr = np.ones((22, 1)).astype(int), values = data, axis = 1)

#her bir kolonun height üzerindeki etkisini ölçmek için bu yapıyı kuruyoruz.
X_list = data_concat.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype = float)
#sm.OLS istatistiksel değerlerimizi çıkarabilmek için kullanıyoruz.
model = sm.OLS(height, X_list).fit()
print(model.summary())

#summary'de x5 in pt değeri en yüksek olduğu için o elenmeli. backwar eli. yöntemine göre.
X_list = data_concat.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list, dtype = float)
model = sm.OLS(height, X_list).fit()
print(model.summary())

#yeni summary'de eski x6 yeni x5 olanın pt değeri 0.03. kabul edilebilir bir değer fakat onu da görmek istemezsek eleyebiliriz.
X_list = data_concat.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list, dtype = float)
model = sm.OLS(height, X_list).fit()
print(model.summary())

""" Code Sectio End"""
