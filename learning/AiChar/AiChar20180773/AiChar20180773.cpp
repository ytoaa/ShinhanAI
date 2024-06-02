// AiChar20180773.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <stdio.h>
#include "MLP.h"

CMLP MultiLayer;

#define NUM_TRAINING_SET 3
#define NUM_INPUT 9
#define NUM_OUTPUT 3
#define NUM_HIDDEN_LAYER 1
#define MAX_EPOCH 100000


int main()
{
    int HiddenNodes[NUM_HIDDEN_LAYER] = { 2 };
    MultiLayer.Create(NUM_INPUT,HiddenNodes,NUM_OUTPUT, NUM_HIDDEN_LAYER);

    // 학습데이터
    int z[NUM_TRAINING_SET][NUM_INPUT] = { {1,1,1,
                                            0,0,1,
                                            0,0,1}, //ㄱ데이터
                                           {1,0,0,
                                            1,0,0,
                                            1,1,1}, //ㄴ 데이터
                                           {1,1,1,
                                            1,0,0,
                                            1,1,1} }; // ㄷ 데이터

    int d[NUM_TRAINING_SET][NUM_OUTPUT] = { {1,0,0}, {0,1,0} ,{0,0,1} };

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
            MultiLayer.pInValue[p+1] = z[n][p];
        MultiLayer.Forward();
        for (int p = 1; p <= NUM_INPUT; p++)
        {
            printf("%.0f", MultiLayer.pInValue[p]);
            if ( p % 3 == 0 && p != NUM_INPUT)
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

            for (int p = 1; p<=NUM_OUTPUT;p++)
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
            if (p % 3 == 0 && p != NUM_INPUT)
                printf("\n");
        }
        printf(" = ");
        for (int p = 1; p <= NUM_OUTPUT; p++)
            printf("%.0f", MultiLayer.pOutValue[p]);
        printf("\n");
    }

    MultiLayer.SaveWeight((char*)"..\\\Weight\\Weight.txt");
    
}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
