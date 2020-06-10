# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:50:24 2020

@author: Gizem Çoban
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
#from keras.datasets import mnist
#from emnist import extract_test_samples
import keras

#kaydedilmiş modeli yükle
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Model Yüklendi")


# Test verilerini yükleme
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test / 255.0
y_test = to_categorical(y_test, 10)
X_test = X_test.reshape((len(X_test), 784))



# Yüklenen modeli test verilerinde değerlendirmek
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Modelin Test Doğruluğu: {0}%'.format(score[1]*100.0))



# 10 tane örneğin test edilmesi
for i in range(10):
    sample = X_test[i].reshape(1, 784)
    prediction = loaded_model.predict_classes(sample)
    target_label = np.argmax(y_test[i])
    print("Tahmin Edilen Rakam: {0}, Gerçek Hedef: {1}".format(prediction[0], target_label))



# Emnist veri seti üzerinde yeni mnist kümesi oluşturma
emnist_images, emnist_labels = extract_test_samples('digits')
print('emnist test görüntüleri şekil: ', emnist_images.shape)
print('emnist test etiketleri şekli: ', emnist_labels.shape)

emnist_images = emnist_images.reshape((len(emnist_images), 784))
emnist_images = emnist_images / 255.0
emnist_labels = to_categorical(emnist_labels, 10)

# modelimizi yeni emnist veri kümesinde değerlendirmek
score = loaded_model.evaluate(emnist_images, emnist_labels, verbose=0)
print('Yüklü modelin EMNIST (genişletilmiş MNIST) veri kümesinde doğruluğu: {0}%'.format(score[1]*100.0))

# ilk 10 emnist test örneğinde sonuçları göster
for i in range(10):
    sample = emnist_images[i].reshape(1, 784)
    prediction = loaded_model.predict_classes(sample)
    target_label = np.argmax(emnist_labels[i])
    print("Tahmin Edilen Rakam: {0}, Gerçek Sonuç: {1}".format(prediction[0], target_label))
    # Resimleri Gösterme
    plt.imshow(sample.reshape(28, 28))
    plt.show()
