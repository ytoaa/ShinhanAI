// AiChar20180773.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <stdio.h>
#include "MLP.h"

CMLP MultiLayer;

#define NUM_TRAINING_SET 10
#define NUM_INPUT 25
#define NUM_OUTPUT 10
#define NUM_HIDDEN_LAYER 4
#define MAX_EPOCH 100000


int main()
{
    int HiddenNodes[NUM_HIDDEN_LAYER] = { 50,40,30,20 };
    MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, NUM_HIDDEN_LAYER);

    // 학습데이터
    // 학습데이터
    int z[NUM_TRAINING_SET][NUM_INPUT] = {
      // 0 데이터
        {1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1},
        // 1 데이터
        {0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1},
        // 2 데이터
        {1, 1, 1, 1, 1,
         0, 0, 0, 1, 1,
         0, 0, 1, 0, 0,
         0, 1, 0, 0, 0,
         1, 1, 1, 1, 1},
        // 3 데이터
        {1, 1, 1, 1, 1,
         0, 0, 0, 1, 1,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         1, 1, 1, 1, 1},
         // 4 데이터
        {1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1},
         // 5 데이터
        {1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         1, 1, 1, 1, 1},
         // 6 데이터
        {1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1},
         // 7 데이터
        {1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1},
         // 8 데이터
        {1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1},
         // 9 데이터
        {1, 1, 1, 1, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 1,
         0, 0, 0, 0, 1,
         0, 0, 0, 0, 1},
    };

    // 레이블
    int d[NUM_TRAINING_SET][NUM_OUTPUT] = {
        // 0
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        // 1
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        // 2
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        // 3
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        // 4
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        // 5
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        // 6
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        // 7
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        // 8
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        // 9
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    };


    if (MultiLayer.LoadWeight((char*)"..\\\Weight\\Weight.txt"))
    {
        printf("기존의 가중치로부터 학습을 시작합니다.\n");
    }
    else
    {
        printf("랜덤 가중치로부터 처음으로 학습합니다. \n");
    }

    //결과출력부

    for (int n = 0; n < NUM_TRAINING_SET; n++)
    {
        for (int p = 0; p < NUM_INPUT; p++)
            MultiLayer.pInValue[p + 1] = z[n][p];
        MultiLayer.Forward();
        for (int p = 1; p <= NUM_INPUT; p++)
        {
            printf("%.0f", MultiLayer.pInValue[p]);
            if (p % 5 == 0 && p != NUM_INPUT)
                printf("\n");
        }
        printf(" = ");
        for (int p = 1; p <= NUM_OUTPUT; p++)
            printf("%.0f", MultiLayer.pOutValue[p]);
        printf("\n");
    }
    getchar();

    // 학습시작

    double MSE;
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++)
    {

        MSE = 0.0;
        for (int n = 0; n < NUM_TRAINING_SET; n++)
        {
            //학습데이터 전달
            for (int p = 0; p < NUM_INPUT; p++)
                MultiLayer.pInValue[p + 1] = z[n][p];

            //러닝
            for (int p = 0; p < NUM_OUTPUT; p++)
                MultiLayer.pCorrectOutValue[p + 1] = d[n][p];

            //출력값 계산

            MultiLayer.Forward();
            //가중치갱신
            MultiLayer.BackPropagationLearning();
            // 갱신이후의 출력 계산
            MultiLayer.Forward();

            for (int p = 1; p <= NUM_OUTPUT; p++)
                MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]) *
                (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);
        }
        MSE /= NUM_TRAINING_SET;
        printf("Epoch%d(MSE)=%.15lf\n", epoch, MSE);
    }

    for (int n = 0; n < NUM_TRAINING_SET; n++)
    {
        for (int p = 0; p < NUM_INPUT; p++)
            MultiLayer.pInValue[p + 1] = z[n][p];
        MultiLayer.Forward();
        for (int p = 1; p <= NUM_INPUT; p++)
        {
            printf("%.0f", MultiLayer.pInValue[p]);
            if (p % 5 == 0 && p != NUM_INPUT)
                printf("\n");
        }
        printf(" = ");
        for (int p = 1; p <= NUM_OUTPUT; p++)
            printf("%.0f", MultiLayer.pOutValue[p]);
        printf("\n");
    }

    MultiLayer.SaveWeight((char*)"..\\\Weight\\Weight.txt");

}
