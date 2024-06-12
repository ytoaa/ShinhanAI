import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.utils import to_categorical # one-hot 인코딩
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)     # 텐서플로우 버전확인 
print(keras.__version__)  # 케라스 버전확인

# MNIST image load (trian, test)
# C:\Users\username\.keras\datasets\mnist.nzp에 다운로드,저장
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data() 
print("load complete")

# 학습데이터와 테스트데이터 설정
# 0~255 중 하나로 표현되는 입력 이미지들의 값을 1 이하가 되도록 정규화    
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# np.expand_dims 차원을 변경
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# label을 ont-hot encoding    
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Sequential 모델 층 구성하기
def create_model():
    model = keras.Sequential() # Sequential 모델 시작
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, 
	activation=tf.nn.relu, padding='same', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPool2D(padding='same'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, 
	activation=tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPool2D(padding='same'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, 
	activation=tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPool2D(padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    return model

model = create_model() # 모델 함수를 model로 변경
model.summary() # 모델에 대한 요약 출력해줌

# CNN 모델 구조 확정하고 컴파일 진행
model.compile(loss='categorical_crossentropy',      # crossentropy loss
              optimizer='adam',                      # adam optimizer
              metrics=['accuracy'])                  # 측정값 : accuracy 

# 학습파라미터 설정
learning_rate = 0.001
training_epochs = 1      #50
batch_size = 100

MODEL_SAVE_FOLDER_PATH = './model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)
model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.keras'
cb_checkpoint = ModelCheckpoint(filepath=model_path, 	monitor='val_loss', verbose=1, 	save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 학습실행
history = model.fit(train_images, train_labels,         # 입력값
     validation_data=(test_images, test_labels),        # test를 val로 사용
     batch_size=batch_size,                                # 1회마다 배치마다 100개 프로세스 
     epochs=training_epochs,                             # 15회 학습
     verbose=1,                               # verbose는 학습 중 출력되는 문구를 설정하는 것 
     callbacks=[cb_checkpoint,cb_early_stopping])    # 중간 결과 확인


# 최종 모델 구조와 가중치 저장
model.save(MODEL_SAVE_FOLDER_PATH+"mnist_model.hdf5")

  
#test 성능평가
score = model.evaluate(test_images, test_labels, verbose=0) # test 값 결과 확인
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 

