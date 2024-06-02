import tensorflow as tf
import function

# 하이퍼파라미터 설정
NUM_INPUT = 5
NUM_HIDDEN = [10, 3]
NUM_OUTPUT = 1
NUM_TRAINING_SET = 30
LEARNING_RATE = 0.01
EPOCHS = 1000

# 학습 데이터 생성
X_train, y_train = function.generate_data(NUM_TRAINING_SET, NUM_INPUT)

# MLP 모델 초기화
mlp_model = function.create_mlp_model(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)

# 옵티마이저 설정
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# 메뉴 표시 및 처리
while True:
    function.display_menu()
    choice = input("메뉴를 선택하세요: ")
    
    if choice == '1':  # 학습 시작
        # 학습 루프
        for epoch in range(EPOCHS):
            loss = function.train_step(mlp_model, optimizer, X_train, y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss: {loss.numpy()}")
    
    elif choice == '2':  # 테스트
        # 테스트 데이터 생성
        X_test, y_test = function.generate_data(5, NUM_INPUT)
        
        # 학습된 모델을 사용하여 예측
        predictions = mlp_model(X_test)
        print("Predictions:")
        print(predictions)
        print("Ground Truth:")
        print(y_test)
    
    elif choice == '3':  # 가중치 저장
        filename = input("가중치 파일 이름을 입력하세요: ")
        if not filename.endswith('.weights.h5'):
            filename += '.weights.h5'  # 파일 이름이 확장자로 끝나지 않으면 추가합니다.
        function.save_weights(mlp_model, filename)
        print("가중치를 저장했습니다.")
    
    elif choice == '4':  # 가중치 불러오기
        filename = input("가중치 파일 이름을 입력하세요: ")
        if not filename.endswith('.weights.h5'):
            filename += '.weights.h5'  # 파일 이름이 확장자로 끝나지 않으면 추가합니다.
        function.load_weights(mlp_model, filename)
        print("가중치를 불러왔습니다.")
    
    elif choice == '5':  # 학습 데이터 생성
        print("학습 데이터를 생성합니다.")
        X_train, y_train = function.generate_data(NUM_TRAINING_SET, NUM_INPUT)
        print("학습 데이터 생성이 완료되었습니다.")
    
    elif choice == '6':  # 종료
        print("프로그램을 종료합니다.")
        break
    
    else:
        print("올바른 메뉴를 선택하세요.")
