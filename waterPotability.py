#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
# Dizilerle çalışmak için kullanılır.Ayrıca doğrusal cebir ve matrisler alanında gerekli işlevlere sahiptir.

import matplotlib.pyplot as plt
# Sayısal matematik hesaplamalarını 2 ya da 3 boyutlu görsel çıktılar olarak almamızı sağlayan bir çizim kütüphanesidir.

import seaborn as sns
# İstatiksel hesaplamalara yeni bir anlam kazandırmak için kullanılan oldukça efektif bir veri görselleştirme kütüphanesidir. 


# # VERİ HAZIRLAMA (DATA PREPARATION)

# In[34]:


data = pd.read_csv(r'C:\Python\water_potability.csv')


# In[35]:


# veri setimizin adını ve adresini belirttik.


# In[36]:


data # veri setimizi ekrana yazdırdık.
data.head()


# In[37]:


data.shape   # ( satır , sütun )


# # Veri Temizleme
# 
# Değersiz verilere(non value/null) sahip miyiz ???

# In[38]:


data.info()


# In[39]:


data.isnull().sum() 
# Kaç adet değersiz(null) veri var ?
# Örneğin sülfat'ta 781 defa değersiz veriye saptanmış. Neredeyse %20 veri kaybı var. (null)


# Çok fazla eksik değer(null) olduğundan bu satırdaki veriler kaldırılabilir.Ama veri sayısı sınırlı durumda! Bu sebeple sütunu(column) yok etmek daha büyük bir veri kaybı olur.

# In[40]:


data.describe()
# Verilerimize ait ortalam, minimum, maksimum vs değerler tanımlanır.


# In[41]:


data


# In[42]:


data = data.fillna(data.mean()) 

# Veri setimizde eksik(non-value) olan yerleri ortalama(mean) değer ile doldurur.

data


# ## Boyutsal anlamda küçültme gerekli midir? Değil midir ? Kontrol edilir.

# In[43]:


sns.heatmap(data.corr())

# Isı haritaları, matris tarzında verileri görselleştirmek için kullanılır.

# corr() yöntemi, veri kümenizdeki her sütun arasındaki ilişkiyi hesaplar. 

"""
plt. show() bir olay döngüsü başlatır, şu anda aktif olan tüm şekil nesnelerini arar ve şeklinizi veya şekillerinizi 
görüntüleyen bir veya daha fazla etkileşimli pencere açar.
"""
plt.show()


# Var olan sıcaklık haritasını daha iyi anlamak için üzerinde biraz oynama yapalım.

# In[44]:


sns.heatmap(data.corr() , annot=True, cmap='CMRmap')
# cmap : Veri değerlerinden renk uzayına eşleme yapmamızı sağlar.


figure = plt.gcf()
# gcf (get the current figure)
# Yeni bir figür oluştur veya mevcut bir figürü etkinleştir.


figure.set_size_inches(14,6)
# Figür boyutu üzerinde değişiklik yapıldı.
plt.show()


# Sonuç : 
# 
# Eğer değerlerimiz arasındaki ilişki %75 veya üzeri olsa idi, benzerlik gösteren iki değişkenden birini çıkartmamız gerekirdi. Çünkü çok yakın veya aynı değerlere sahip değişkenler model eğitimi için gereksiz yani istenmeyen bir durumdur.Ama bizim değişkenlerimiz arasındaki ilişki için böylesi bir durum söz konusu değildir.

# ## Kutu Grafiğini Kullanarak Aykırı Değeri Kontrol Edelim

# In[45]:


data.boxplot(figsize = (15,6))

# boxplot : Data sütunlarından bir kutu grafiği yapın.
# figsize : Matplotlib'de oluşturulacak şeklin boyutu.

plt.show()


# In[46]:


data['Solids'].describe()


# 
# Bu tarz aykırı değerler veriler içersinde olabilir. Burada bu verinin kaldırılıp, kaldırılmaması üzerine düşünülmesi 
# gerekir. Bu aykırı elementin kaldırılması bizim modelimizin eğitimi için doğru olmaz ve içilebilirliğin sürekli olumlu olmasını
# sağlar. Bu da modelimizi önyargılı hale getirebilir.
# 
# Diğer bir seçenek olarak ise modelimizi bu verimizle ve de bu verimizi kaldırarak denememiz sonucu yapılan tahminin doğruluk oranları (accuracy score) arasında değerlendirme yapıp hangi durumun daha bizim için daha doğru olduğunu anlayabiliriz.
# 
# 

# ### Suyun kalitesine karar vermede önemli olabileceği için aykırı değerleri çıkarmıyoruz!

# ------------------------------------------------------------------------------------------------------

# # Modelimizin son haline kısa bir göz atalım.

# In[47]:


data.head()


# In[48]:


data.shape


# In[49]:


data.info()


# In[50]:


data.isnull().sum()


# In[51]:


data['Potability'].value_counts()


# In[52]:


sns.countplot(data['Potability'])
plt.show()


# Hedef değişkenimiz arasındaki denge durumu iyidir. Bu durumun dengesiz (unbalanced) olması modelimizi önyargılı yapar ve olumsuz sonuçlar doğurur.

# ---------------------------------------------------------------------------------------------------------------------------

# In[53]:


data.hist(figsize = (14,12))
plt.show()


# Graflarımız normal olduğundan ötürü normalleştirmeye(normalization) ihtiyacımız yoktur.

# ## Bölümleme

# In[54]:


x = data.drop('Potability', axis=1) 

# Hedef değişkeni aradan çıkar.


# In[55]:


x


# In[56]:


y = data['Potability'] # Hedef değişkenimizi 'y' diye tanımladık.


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split


# In[292]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,shuffle=True , random_state=20)


# In[293]:


# shuffle(list) Verdiğiniz bir liste içindeki değerlerin sırasını karıştırır.

"""
random_state parametresi, sizin durumunuzda verilerin eğitim ve test endekslerine bölünmesine karar verecek olan 
dahili rasgele sayı üretecini başlatmak için kullanılır.

"""
x_train


# In[294]:


y_train


# # MODEL EĞİTME (MODEL TRAINING) 

# ## Karar Ağacı (Decision Tree)

# In[295]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[296]:


dt. fit(x_train,y_train)


# In[297]:


x_test


# In[298]:


y_test


# In[299]:


y_prediction=dt.predict(x_test) # Burada modelimizi denemek istiyoruz. Bakalım doğru tahmin edebilecek mi ?

dt.predict(x_test)


# In[300]:


from sklearn.metrics import accuracy_score, confusion_matrix

# accuracy_score -> Doğruluk sınıflandırma puanını ortaya koyar.


# In[301]:


accuracy_score(y_prediction,y_test) * 100 # Tahmin başarı yüzdesini görebilmek için 100 ile çarptık.


# In[302]:


print(confusion_matrix(y_prediction,y_test))


# In[241]:


print(y_test.shape)


# ### Eğittimiz modelimizin başarı oranının çok iyi oranda olmadığını gözlemledik ? !!!

# Ama bu tahmin üzerine yapılan başarı tabiri çalışılan alana göre değişiklik gösterir. Örneğin kendi kendine sürüş özelliği 
# olan bir aracın başarı oranının %95 olması bir başarı değil aksine başarısızlıktır.Aracın içesinde birisi olduğunu düşünür isek ve burada %95 gibi bir oranla sürüş başarına sahip olsak dahi insanın can sağlığı söz konusu olduğu ve olası bir hatanın ölüm ya da ölümlerle sonuçlabilme ihtimali mümkündür. Bu sebeple %100 başarı oranına sahip olunması gerekir. Fakat psikoloji, nöroloji vb. depresif alanlardaki %50 ve üzeri başarı oranı veya doğru tespitler sevindiricidir. Çünkü doktor dahi hastalık tespitinde çok yüksek bir başarı oranına sahip değildir ve %50 üzeri doğru tahminler için doktor başarılır diyebiliriz. Bundan dolayı doğruluk oranını değerlendirirken hangi alan olduğu önem teşkil etmektedir. Buradan yola çıkarak çalışmamızın pek de başarısız ve yetersiz olmadığını düşünebiliriz.
