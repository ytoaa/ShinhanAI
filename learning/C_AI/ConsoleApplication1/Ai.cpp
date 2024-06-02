// ConsoleApplication1.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include "MLP.h"

CMLP MultiLayer;
#define PI 3.14159265359
#define MAX_EPOCH 100000
#define NUM_INPUT 5
#define NUM_OUTPUT 1
#define NUM_HIDDEN_LAYER 2
#define NUM_TRAINING_SET 30
void DisplayMenu();
void GenerateData();
void LearnihgStart();
void LoadWeight();
void SaveWeight();
void test();

int main()
{
    DisplayMenu();
    
    char sh;
    int HiddenNodes[NUM_HIDDEN_LAYER] = { 10,3 };
    MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, NUM_HIDDEN_LAYER);
    while ((sh = getchar()) != EOF) {
        switch (sh)
        {
        case '1':
            printf("학습시작");
            LearnihgStart();
            break;
        case '2':
            printf(" weight 저장");
            SaveWeight();
            break;
        case '3':
            printf(" weight 로드");
            LoadWeight();
            break;
        case '4':
            printf("test4");
            test();
            break;
        case '5':
            printf("데이터 생성 시작");
            GenerateData();
            break;
        case '6':
            printf("test6");
            break;
        case '0':
            return 0;
        }
        printf("\n");
        DisplayMenu();
    }


}

void DisplayMenu() {
    
    char manu[6][64] = { "[1] 학습시작","[2] weight 저장","[3] weight 읽기","[4] test","[5] 학습데이터 생성","[6] 중단" };
    for (int i = 0; i < 6; i++) {
        printf("%s\n", manu[i]);
    }
    printf("\n선택매뉴");
}

void LearnihgStart() {
 
    int i,j,p;
    double z[NUM_TRAINING_SET][NUM_INPUT] = {};

    double d[NUM_TRAINING_SET][NUM_OUTPUT] = {};
    FILE* fp = fopen("TrainingData.txt", "rt");
    if (fp == NULL) {
        printf("파일을 읽을수 없습니다.");
        return;
    }
    for (i = 0; i < NUM_TRAINING_SET; i++) {
        for (j = 0; j < NUM_INPUT; j++) {
            fscanf(fp, "%lf", &z[i][j]);
        }
        fscanf(fp, "%lf", &d[i][0]);
    }
    fclose(fp);

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
    printf("학습 종료\n");
}

void GenerateData() {
    float value = 0;
    float nextvalue = 0;
    int set, j;
    FILE* fp = fopen("TrainingData.txt", "wt");
    for ( set = 0; set < NUM_TRAINING_SET; set++)
    {
        for ( j = NUM_INPUT ; j > 0 ; j--)
            value = sin(2 * PI / NUM_TRAINING_SET * (set-j)) / 2 + 0.5; //0~1
        fprintf(fp, "%f", value);
        value = sin(2 * PI / NUM_TRAINING_SET * (set - j)) / 2 + 0.5;
        fprintf(fp, "%f", nextvalue);
    }
    fclose(fp);
}

void SaveWeight() {
    MultiLayer.SaveWeight((char*)"Weight.txt");
    printf("가중치를 저장했습니다");
}

void LoadWeight() {
    if (MultiLayer.LoadWeight((char*)"Weight.txt")) {
        printf("가중치를 불러왔습니다.");
    }
    else
        printf("가중치 불러오기를 실패했습니다.");
    
}

#include <stdlib.h>
void test()
{
    int i;
    double value=0;
    double offset=0;
    srand(time(NULL));
    offset = (double)rand()/RAND_MAX * 2 * PI;
    for (i = 0; i < NUM_INPUT; i++)
    {
        value= sin(2 * PI / NUM_TRAINING_SET * (i)+offset) / 2 + 0.5;
        MultiLayer.pInValue[i+1] = value;
    }
    value = sin(2 * PI / NUM_TRAINING_SET * (i)+offset) / 2 + 0.5;
    MultiLayer.Forward();
    printf("\n [출력] %lf,%lf,%lf,%lf,%lf=%lf(%lf)",
        MultiLayer.pInValue[1], MultiLayer.pInValue[2], MultiLayer.pInValue[3], MultiLayer.pInValue[4], MultiLayer.pInValue[5],
        MultiLayer.pOutValue[1],value);
}