// AiChar20180773.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <stdio.h>
#include "MLP.h"

CMLP MultiLayer;


#define NUM_INPUT 25
#define NUM_OUTPUT 10
#define NUM_HIDDEN_LAYER 4



int main()
{

    printf("20180773 유우식.\n");
    int HiddenNodes[NUM_HIDDEN_LAYER] = { 50,40,30,20 };
    MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, NUM_HIDDEN_LAYER);

    // 학습데이터


    if (MultiLayer.LoadWeight((char*)"..\\\Weight\\Weight.txt"))
    {
        printf("기존의 가중치로부터 학습을 시작합니다.\n");
    }
    else
    {
        printf("가중치를 읽을수 없습니다.. \n");
        return 0;
    }

    int test_input[NUM_INPUT] = { 1,1,1,1,1,
                                  1,0,1,0,1,
                                  1,1,1,1,1,
                                  1,0,1,0,1,
                                  1,1,1,1,1};

    for (int p = 0; p < NUM_INPUT; p++)
        MultiLayer.pInValue[p + 1] = test_input[p];

    MultiLayer.Forward();


    for (int p = 1; p <= NUM_INPUT; p++)
    {
        printf("%.0f", MultiLayer.pInValue[p]);
        if (p % 5 == 0 && p != NUM_INPUT)
            printf("\n");
    }
    printf("\n");
    for (int p = 1; p <= NUM_OUTPUT; p++)
        printf("%d => %.3f%%  \n",p-1, MultiLayer.pOutValue[p]*100);
    printf("\n");

}

