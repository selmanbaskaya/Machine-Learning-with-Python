"""
@author: Selman
"""

""" Libraries """

"""
Kütüphanelerin Yüklenmesi
as pd veya as plt yazarak import edilen kütüphaneleri ileride daha kısa isimler ile kullanabiliyor olacağız.
"""
import pandas as pd
""" Libraries End"""

""" Code Section """

"""
Veri Ön İşleme

Veri Setinin Yüklenmesi
Veri seti kodlarımızı yazdığımız dosya ile aynı dizinde ise direkt olarak veri setinin ismini yazarak
yükleme işlemini gerçekleştirebiliriz.
data = pd.read_csv('sales.csv')
"""

data = pd.read_csv('datasets/sales.csv')
print(data)
"""
Eğer ki veri setimiz kod dosyamız ile aynı dizinde değilse aşağıdaki adımları takip ederek veri setimizi
yükleyebiliriz (Her işletim sistemi için path'in yazımı aşağıda gösterilmiştir);

data = pd.read_csv('Users/username/.../file_name.csv') (Mac)
data = pd.read_csv('C:\\Users\...\file_name.csv') (Windows)
data = pd.read_csv('home/users/.../file_name.csv') (Linux)


pandas'ın dataframe özelliğinden yararlanarak data değişkeninin içerisinden istediğimiz bir kolunun bilgilerini
farklı bir değişken içerisine çekebiliriz.
"""
months = data[['Months']]
print(months)

sales = data[['Sales']]
print(sales)
"""
months ve sales kolonlarını tek bir değişken içerisinde tutmak da tabii ki mümkün;
months_sales = data[['Months', 'Sales']] şeklinde yazarak tek bir değişken ile de verileri tutabiliriz.

Kolondaki verilerimizi array değil de dataframe formatında tutmak istersek iloc komutunu kullanmamız gerekiyor.
"""
sales_2 = data.iloc[:,1:].values
print(sales_2)

#Veri Setinin Train ve Test Olarak Bölünmesi
from sklearn.model_selection import train_test_split
"""
Veri setimizi bölmek için 4 değişkene ihtiyacımız var;
x_train ve y_train değişkenlerini eğitim için x_test ve y_test değişkenlerini eğitim sonunda model'i test etmek
için kullanıyor olacağız. x'ler bizim için bağımsız değişken iken y'ler x'lere bağlı değişkenlerdir.
veri setini bölmek için kullanılan ilk parametre bağımsız değişkenlerimizi ikinci parametre ise bağımlı değişkenlerimizi
temsil etmektedir. test_size parametresi ise veri setinin yüzde kaçını test için ayıracağımızı belirtir, son olarak
random_state parametresi ise kodları tekrar tekrar çalıştırdığımızda bölünen verilerin sabit kalması için kullanılır.
"""
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33, random_state = 0)

"""
Özellik Ölçeklendirme
Verilerin birbirlerine olan uzaklık değerlerini daha standart bir hale getirmek için kullanıyoruz. 
StandartScaler işleminde önce Months kolununda 11,25,34,54,64 gibi birbirinden uzak değerler vardı
StandartScaler ile bu değerleri -1.6 ve 1.3 arasında kadar düşürmüş olduk.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#Model Oluşturma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#x_train ve y_train verilerine bakarak bir ilişki kurmaya çalışıyoruz. Bunun içinde fit fonksiyonunu kullanıyoruz.
lr.fit(X_train, Y_train)

'''
predict fonksiyonuna x_test verilerini yollayarak şuana kadar öğrendiği kadarı ile bir tahmin yapmasını istiyoruz.
Sizinde fark ettiğiniz gibi predict yaparken hiçbir şekilde y_test verisini vermiyoruz, amacımız modelin train setleri
ile öğrendiklerini x_test üzerinde uygulayarak tahminler üretmesii.
print fonksiyonu ile y_pred ve y_test değişkenlerini ekrana yazdırıp modelimizin ne kadar başarılı olduğuna bakabiliriz
'''
y_pred = lr.predict(X_test)
print("Tahmin edilenler - StandartScaler Uygulanan")
print(y_pred)
print("Gerçek değerler - StandartScaler Uygulanan")
print(Y_test)

'''
Yukarıda aldığımız sonuçlar StandartScaler'dan geçtiği için belki sizler için anlamlı gelmeyebilir, normal değerler
üzerinden predict işlemlerini aşağıdaki şekilde yapabiliriz.
'''
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("Tahmin edilenler - StandartScaler Uygulanmayan")
print(y_pred)
print("Gerçek değerler - StandartScaler Uygulanmayan")
print(y_test)

'''
Görselleştirme
Verileri görselleştirmeden önce grafiğin anlaşılabilir olması adına verileri sort_index ile sıralıyoruz. Sizler 
çalışırken her iki türlü de deneyerek ne demek istediğimi çok daha iyi anlamış olacaksınız.
plot fonksiyonuna x ve y train verilerimizi vererek aylık satış dağılımlarını grafik üzerinden görebiliyoruz.
plot fonksiyonuna x ve predict x test verilerimizi vererek modelimizin doğrusunu görmüş oluruz.
grafiğimize title ile grafik başlığı, xlabel ile x eksenini başlığı ve ylabel ile de y ekseni başlığını eklemiş oluruz.
'''
import matplotlib.pyplot as plt

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.title("sales by month")
plt.xlabel("months")
plt.ylabel("sales")

""" Code Sectio End"""
