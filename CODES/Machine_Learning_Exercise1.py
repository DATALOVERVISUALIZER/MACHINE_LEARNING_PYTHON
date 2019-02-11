# herhangi bir seyin davranisina dair pattern'leri bulup bu pattern'lerle o seyin gelecegini tahmin etmek .

'''1950'lerden itibaren gelişmeye başlamıştır.
1970-80 arası machine learning için kara kış olarak geçer.
80 sonrası donanımda ilerlemeler ile insanlar daha fazla data tutabilir hala geliyorlar.
supervised learning: gender classficiation (insanların boyuna, ayak boyuna gibi bilgileri ile bir model geliştirilebilir)
 ** kritik nokta data seti ve data setining hangi sınıfa ati olduğu bellidir.
 **Training ve test olarak ikiye ayrılır.
training seti gerekli label'lar ile test edilir.
temel olarak ikiye ayrılır: Classification ve Regression.(reg numeric değerlerden tahmin edilir,ist'da evlerin fiyatlarını tahmin eden değer)
unsupervised  learning:label yoktur. Etiket yoktur.Süpermarket müşterilerinin gruplanması olabilir. Clustering kümeleme örnek olarak verilebilir.
**CLUSTERING: iki küme için başarılı şekilde ayrıştığında dataya daha fazla girip gelir dağılımı nasıl şeklinde yorumlar yapılabilir. Daha çok customer segmentationda kullanılılır. Müşterilerin davranış pattern'ları incelenebilir.
**DIMENSINABILITY:Boyut indirgeme



White BOX:Bankacılık da white box olması oldukca önemlidir. Algoritmanın açıklanabilir olması gereklidir.

BLACK BOX:input verilir iyi output da verebilir ancak içerdeki süreç bilinmeyebilir. ÖZellikle yapay sinir ağları için durum geçerlidir. süreç anlanmak da zorlanılabilir. Çok derin bir matematiksel süreç geçiyor olabilir.


feature vector:ayakkabı numarası, boy, kilo(feature selection)

dimensinability Redection: Eksen indirgeme: Temel amaç, feature'ların özniteliklerin azaltılmasıdır.Hız artar, Visualization için de önemlidir.Visuzalization kolaşlaştırır.


**************************************************************************TASK NUMBER1:WHAT IS MACHINE LEARNING************************************************************
 Makine Öğrenmesi (Machine Learning), matematiksel ve istatistiksel yöntemler kullanarak mevcut verilerden çıkarımlar yapan, bu çıkarımlarla bilinmeyene dair tahminlerde bulunan yöntem paradigmasıdır
Makine öğrenmesine güncel hayatımızdan bazı örnekler: yüz tanıma, belge sınıflandırma, spam tespiti.
Gözlemler (Observations): öğrenmek ya da değerlendirmek için kullanılan her bir veri parçası. Örn: her bir e-posta bir gözlemdir.

Özellikler (Features): Bir gözlemi temsil eden (genelde sayısal) verilerdir. Örn: e-posta'nın uzunluğu, tarihi, bazı kelimelerin varlığı.

Etiketler (Labels): Gözlemlere atfedilen kategoriler. Örn: spam, spam-değil.

Eğitim Verisi (Training Data): Algoritmanın öğrenmesi için sunulan gözlemler dizisi. Algoritma bu veriye bakarak çıkarımlarda bulunur, kafasında model kurar. Örn: çok sayıda spam/spam-değil diye etiketlenmiş e-posta gözlemi

Test Verisi (Test Data): Algoritmanın kafasında şekillendirdiği modelin ne kadar gerçeğe yakın olduğunu test etmek için kullanılan veri seti. Eğitim esnasında saklanır, eğitim bittikten sonra etiketsiz olarak algoritmaya verilerek algoritmanın (vermediğimiz etiketler hakkında) tahminlerde bulunması beklenir. Örn: spam olup olmadığı bilinen (ama gizlenen), eğitim verisindekilerden farklı çok sayıda e-posta gözlemi

SUPERVISED LEARNING(Gözetimli Öğrenme)
Etiketlenmiş gözlemlerden öğrenme sürecidir. Etiketler, algoritmaya gözlemleri nasıl etiketlemesi gerektiğini öğretir. Örneğin içinde "para kazan" ifadesi geçiyorsa spam demelisin gibi yol göstermelerde bulunur.

Sınıflandırma (Classification): Her bir gözleme bir kategori/sınıf atması yapar: Örn: spam/spam değil. Sınıflar ayrıktır (sayı değildir) ve birbirlerine yakın/uzak olmaları gibi bir durum söz konusu değildir.

Regresyon (Regression): Her gözlem için öğrendiklerine bakarak reel bir değer tahmini yapar. Örn: "2011 model 40.000 km'de Mia Torento arabanın fiyatı 45.670 TL olmalıdır".

UNSUPERVISED LEARNING:(Gözetimsiz Öğrenme)

Etiketsiz gözlemlerden öğrenme sürecidir. Algoritmanın kendi kendine keşifler yapması, gözükmeyen örüntüleri keşfetmesi beklenir.

Kümeleme (Clustering): Gözlemleri homojen bölgelere ayırır. Örn: bir okuldaki öğrenci gruplarını tespit etmek.

Boyut Azaltımı (Dimensionality Reduction): Gözlemlerin mevcut özellik sayısını az ve öz hale indirir, bize en iyi öğrenme imkanı sunar.Makina öğrenmesinin önceden yönlednirilmesine dayanır. Makina gözlem ve şartlara göre öğrenir.

Süpervised vs unsupervised:spam mailing örneğin supervised yöntemdir. Datanın label'ları vardır. Classfication yöntemi kullanılabilir.
PTE (pearson test of english) supervised learning , grammer hataları vb önceden etiketlenmiş olabilir. Supervised learning yöntemidir genel olarak.




******************************************************TASK2:SUPERVISED VS UNSUPERVISED*********************************************************
SUPERVISED(classification,regression
**supervised label'lar bilinir. classfication datada kolaylıkla yapılabilir.
Classification output labels yaparken, regregssion countionus bir sonuc verir rakamlardan.
classification da da regression'da da amaç input datada specific bir ilişki bulmaktır.
classification exp:Spam Detection, Churn Prediction, Sentiment Analysis,Dog Breed Detection.
Regression predicts a numerical value based on previous observed data.
eg: House Price Prediction, Stock Price Prediction, Height-Weight Prediction.


Unsupervised(clustering,dimensionability reduction)

clustering:dataların benzerliklerine göre sınıflandırılıp kümelendirilmedir.Bu data pointleri arasındaki relationshipler incelenebilir.

**Here is a list of some unsupervised machine learning algorithms:

K-means clustering
Dimensionality Reduction
Neural networks / Deep Learning
Principal Component Analysis
Singular Value Decomposition
Independent Component Analysis
Distribution models
Hierarchical clustering

ODEABANK'ta yapılan bir projede hedef müşteri listesi, ilgili iş birimlerinden alındıktan sonra ilgili haberler tüm sitelerde taranıp, müşteri hakkında çıkan haberler positive, negative, neutral olarak
gruplanıp ilgili iş birimlerine mail atılır.Unsupervised yöntem kullanılmıştır.



********************************************TASK 3 DATA PROCESSING STEPS*****************************************************************************************
null değerler için replacement methods:
1. mean yazılabilir.(normal dağılım ise mean değeri yazılabilir. Temsili bir değer dağlar)
Normal dağılım göstermiyorsa, mean değeri null values yerine kullanılamaz. Meadian (orta nokta) değeri yazılabilir.

Data manupulaton: pivot olan datalar column, row değerlere donüştürülmek zorunda kalınabilir. Null değerlerin replace edilmesi.
SCALING:min max scaling, z test,
Data cleaning: Outlier'ların temizlenmesi.Noise olan , outlier olan dataların temizlenmesi.

BIR DATA SCIENCE PROJESI ICIN steps:
1.seeting the search goal,
2.preprossesgin: Normalization, data cleaning, data manupulation, scale edilmeli(parametre cok buyuk sayıysa modelı yanlıs yonlendirebilir)SCALING:veri setini ölçeklendirmek, Örneğin bir kolon: 80,82,89,92 değerlerine sahipken, diğer kolon 1.3,1.7,2.1 değerlerine sahip ise scaling yapılmalı(scaling yöntemleri, median, z score,standardzation, min/max bitirme projesi için standarsization önemli teşkil etmektedir)
outliers
noveltes
noise
deviations
exceptions


'''
#Example use standart scaling and min max  methods
# normalization  min max
# standardizing is scaling
#  MIN MAX NORMALIZATION FORMULA
# Normalization is used to scale the data between 0 and 1. It is defined as
#
# Yi = [Xi - min(X)]/[max(X) - min(X)]

# The formula for Z-SCORE normalization is below:
#
# \frac{value - \mu}{\sigma}
# σ
# value−μ
# ​

# SUMMARY
# Min-max normalization: Guarantees all features will have the exact same scale but does not handle outliers well.
# Z-score normalization: Handles outliers, but does not produce normalized data with the exact same scale.


import warnings
import pandas as pd
import math
warnings.filterwarnings('ignore')


iris = pd.read_csv("iris.csv")
iris.columns
iris.head()

iris.head()
iris.tail()


# SCALING
# Rescale data (between 0 and 1)
# Normalize the data attributes for the Iris dataset.
# Code ref: https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/

from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
y = iris.target
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# ikinci yöntem
print(preprocessing.normalize(X))
# The smallest value becomes the 0 value and the largest value becomes 1. All other values fit in between 0 and 1.


# sndardize the data attributes for the Iris datasetPython

# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the Iris dataset
iris = load_iris()
# separate the data and target attributes
X = iris.data
y = iris.target
# standardize the data attributes
standardized_X = preprocessing.scale(X)
1
2
3
4
5
6
7
8
9
10
11
# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the Iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data and target attributes
X = iris.data
y = iris.target
# standardize the data attributes
standardized_X = preprocessing.scale(X)

# SECOND LONG WAY DO IT WITH PROCEDURE
# Example Data
from sklearn.datasets import load_iris

# load the Iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data and target attributes
X = iris.data
y = iris.target
print(iris.data)

iris.head()

# #z score
normalized_df=(iris['sepal.length']-iris['sepal.length'].mean())/iris['sepal.length'].std()
 print(normalized_df)

 # min max
 min_max_normalized_df=(iris['sepal.length']-iris['sepal.length'].min())/(iris['sepal.length'].max()-iris['sepal.length'].min())
print(min_max_normalized_df)



