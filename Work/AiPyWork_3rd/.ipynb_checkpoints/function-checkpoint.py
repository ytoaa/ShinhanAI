# function.py

import tensorflow as tf
import numpy as np

# 모델 가중치 저장 함수 정의
def save_weights(model, filename):
    model.save_weights(filename)

# 모델 가중치 불러오기 함수 정의
def load_weights(model, filename):
    model.load_weights(filename)

# MLP 모델 정의
def create_mlp_model(input_size, hidden_sizes, output_size):
    model = tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(tf.keras.layers.Dense(hidden_size, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(output_size))
    return model

# 학습 함수 정의
@tf.function
def train_step(model, optimizer, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.square(predictions - targets))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 학습 데이터 생성 함수 정의
def generate_data(num_training_set, num_input):
    X = np.random.rand(num_training_set, num_input)
    y = np.sin(2 * np.pi * X).sum(axis=1, keepdims=True) / 2 + 0.5
    return X.astype(np.float32), y.astype(np.float32)

# 메뉴 표시 함수 정의
def display_menu():
    print("[1] 학습 시작")
    print("[2] 테스트")
    print("[3] 가중치 저장")
    print("[4] 가중치 불러오기")
    print("[5] 학습데이터 생성")
    print("[6] 종료")
