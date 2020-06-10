# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:50:24 2020

@author: Gizem Çoban
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.optimizers import Adam
#çoklu sınıflı veri kümeleri nedeniyle categorical kullanıldı
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)

#mnist veri kümesinden 6000 görüntüleri import etme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape[0])

#Herhangi bir hata oluştuğunda onu göstermek için 
assert(X_train.shape[0] == y_train.shape[0]), "Görüntü sayısı eşit değil. .."
assert(X_test.shape[0] == y_test.shape[0]), "Görüntü sayısı eşit değil. .."
assert(X_train.shape[1:] == (28, 28)), "Görüntülerin boyutu 28x28 değil"
assert(X_test.shape[1:] == (28, 28)), "Görüntülerin boyutu 28x28 değil"


#Çıktı değerlerinin kategorileştirilmesi
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Her görüntünün Yoğunluğu 0 ila 255 arasındadır
X_train = X_train/255
X_test = X_test/255

# resimlerin şeklini 1d dizisine değiştirmeliyiz (28 * 28)
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0],
                         num_pixels)
X_test = X_test.reshape(X_test.shape[0],
                         num_pixels)
print(X_train.shape)

#YSA Modelinin Oluşturulması
#İlk katman (girdi katmanı) 10 tane nöron olsun, 784 tane özelliğin olması ve relu aktivasyon fonk.

def create_model():
  model = Sequential()
  model.add(Dense(10, input_dim = num_pixels,
                  activation = 'relu'))
  model.add(Dense(30, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  model.compile(Adam(lr=0.01),
                loss='categorical_crossentropy',
               metrics=['accuracy'])
  return model

model = create_model()
print(model.summary())

#Modelin eğitilmesi
history = model.fit(X_train, y_train, validation_split=0.1,
         epochs=15, batch_size=200, verbose=1, shuffle=1)

#Kayıp ve Başarıyı grafikte gösterme
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')

'''plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')'''

#Tahmin yapılan kısım
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


#Modeli Kaydetme
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#ağırlıkları HDF5'e seri hale getirme
model.save_weights("model.h5")
print("Model kaydedildi")

