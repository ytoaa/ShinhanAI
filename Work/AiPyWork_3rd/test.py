import tensorflow as tf

# 모델 저장 함수 정의
def save_model(model, filename):
    model.save(filename + '.keras')

# 모델 불러오기 함수 정의
def load_model(filename):
    return tf.keras.models.load_model(filename + '.keras')

# 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 저장
save_model(model, 'my_model')

# 모델 불러오기
loaded_model = load_model('my_model')

# 모델 요약 정보 출력
loaded_model.summary()
