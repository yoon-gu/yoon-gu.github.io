---
layout: post
title:  "Transfer Learning"
author: Y Hwang
date:   2021-06-11 15:55:10 +0900
categories: tutorials
tags: [Transfer Learning, 전이학습, CNN, Tensorflow, MobileNetV2]
---

### Notice ###
1. 이 글은 글쓴이는 `현재의 나`, 대상 독자는 `(다 까먹은) 미래의 나`입니다.
1. 논리적인 설명은 지양하고, 시간 순서대로 저의 삽질을 적어놓았습니다.


### What you will get ###

1. `tensorflow`를 사용하여 Transfer Learning을 구현할 수 있습니다. (성능 향상을 고려하지 않았습니다.)
1. 제가 얼마나 삽질을 엄하게 하는지 알 수 있습니다.

### 학습 폴더를 바로 Generator로 만들기 ###
```python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import numpy as np

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

# 학습용
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_dataset = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=IMG_SIZE)

# 검증용
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_dataset = validation_datagen.flow_from_directory(validation_dir,
                                                              shuffle=True,
                                                              batch_size=BATCH_SIZE,
                                                              target_size=IMG_SIZE)
```

### 사전 학습 된 모델 불러오기 ###
다음 코드를 통해 간단하게 사전학습 된 MobileNetV2 모델을 로드할 수 있습니다.
```python
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
```
`base_model.summary()`를 하면 다음과 같이 Prediction 부분이 짤린 모델구조를 볼 수 있습니다.
```
Model: "mobilenetv2_1.00_160"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
__________________________________________________________________________________________________
...
__________________________________________________________________________________________________
Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
==================================================================================================
Total params: 2,257,984
Trainable params: 0
Non-trainable params: 2,257,984
__________________________________________________________________________________________________
```

`include_top=True`로 설정하면 다음과 같이 마지막 레이어까지 확인할 수 있습니다. Transfer Learning은 이렇게 최종 레이어까지 로드할 일은 없습니다.
```
Model: "mobilenetv2_1.00_160"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
__________________________________________________________________________________________________
...
__________________________________________________________________________________________________
Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 1280)         0           out_relu[0][0]                   
__________________________________________________________________________________________________
predictions (Dense)             (None, 1000)         1281000     global_average_pooling2d_1[0][0] 
==================================================================================================
```

### 모델 구성하기 ###

여기서는 Functional API를 사용하는 것이 좋습니다. 아래 코드를 통해 사전학습 된 모델은 더 이상 학습하지 않도록 설정합니다.
```python
base_model.trainable = False
```

```python
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
```

Functional API 방식으로 모델을 구성할 때는 반드시, `inputs`과 `outputs`가 Tensorflow 연산으로 연결되어 있어야 합니다.
제가 자주 하는 실수는 다음 2가지가 있습니다. 첫째는 `Model(inputs, base_model)`로 선언하는 것입니다.
```python
model = tf.keras.Model(inputs, base_model) # Error 1
```
그러면 다음과 같이 나옵니다.
```
AttributeError: 'Model' object has no attribute 'op'
```
`base_model`은 모델이지 Tensor가 아니기 때문입니다.
저는 반복적으로(...) 이러한 사실을 깨닫고, "아차, `base_model의` 최종 출력을 넣어야지!"를 마음속으로 외칩니다.
제가 자주 하는 두번째 실수가 여기서 나옵니다.
```python
model = tf.keras.Model(inputs, base_model.output) # Error 2
```
이 경우 다음과 같이 그래프가 끊어졌다고 나옵니다.
```
ValueError: Graph disconnected: cannot obtain value for tensor Tensor("input_3:0", shape=(None, 160, 160, 3), dtype=float32) at layer "input_3". The following previous layers were accessed without issue: []
```

정신을 차리고 "연결"이라는 단어를 생각합니다. "연결"이 되기 위해서는 2가지가 존재해야합니다. 여기서는 `Model()`에 들어가는 2가지 텐서를 말합니다.
2개의 텐서가 Tensorflow 연산으로 연결되어 있지 않으면, Forward Pass와 Backpropagation을 자동으로 계산할 수 없습니다.

다음과 같이 적용해야 합니다.
```python
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

### 학습하기 ###
학습을 하면 됩니다.
```python
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
```

### 지금 생각난 TODO들###
1. `tf.keras.Sequential`를 이용해서 모델을 구성할 수는 없을까요?
1. 사전 학습된 모델을 전체 학습시키는 방법
1. Data Augmentation하는 방법

### Code ###
```python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import numpy as np

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

# 학습용
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_dataset = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=IMG_SIZE)

# 검증용
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_dataset = validation_datagen.flow_from_directory(validation_dir,
                                                              shuffle=True,
                                                              batch_size=BATCH_SIZE,
                                                              target_size=IMG_SIZE)
# MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False # MobileNetV2의 Variable들 업데이트 하지 않도록 설정

# 새로운 모델 구성
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)


model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```