#include <stdio.h>
#include "MLP.h"
CMLP MultiLayer;

int main()
{
	int HiddenNodes[2] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(2, HiddenNodes, 1, numofHiddenLayer);

	double x[4][2] = { {0,0},{ 0,1 },{ 1,0 },{ 1,1 } };
	//test

	MultiLayer.m_Weight[0][0][1] = -0.8;
	MultiLayer.m_Weight[0][1][1] = 0.5;
	MultiLayer.m_Weight[0][2][1] = 0.4;

	MultiLayer.m_Weight[0][0][2] = 0.1;
	MultiLayer.m_Weight[0][1][2] = 0.9;
	MultiLayer.m_Weight[0][2][2] = 1.0;

	MultiLayer.m_Weight[1][0][1] = -0.3;
	MultiLayer.m_Weight[1][1][1] = -1.2;
	MultiLayer.m_Weight[1][2][1] = 1.1;
	MultiLayer.pInValue[1] = 1;
	MultiLayer.pInValue[2] = 1;

	MultiLayer.pCorrectOutValue[1] = 0;

	MultiLayer.Forward();
	MultiLayer.BackPropagationLearning();

	printf("\nWeights after learning:\n");
	for (int layer = 0; layer < MultiLayer.m_iNumTotalLayer - 1; ++layer) {
		printf("Layer %d to Layer %d:\n", layer, layer + 1);
		for (int snode = 0; snode <= MultiLayer.m_NumNodes[layer]; ++snode) {
			for (int enode = 1; enode <= MultiLayer.m_NumNodes[layer + 1]; ++enode) {
				printf("Weight[%d][%d][%d]: %lf\n", layer, snode, enode, MultiLayer.m_Weight[layer][snode][enode]);
			}
		}
	}

	// Print node-wise output values after learning
	printf("Node-wise output values after learning:\n");
	for (int layer = 0; layer < MultiLayer.m_iNumTotalLayer; ++layer) {
		printf("Layer %d:\n", layer);
		for (int node = 1; node <= MultiLayer.m_NumNodes[layer]; ++node) {
			printf("Node %d: %lf\n", node, MultiLayer.m_NodeOut[layer][node]);
		}
	}

	// Print error gradient values after learning
	printf("\nError gradient values after learning:\n");
	for (int layer = 0; layer < MultiLayer.m_iNumTotalLayer; ++layer) {
		printf("Layer %d:\n", layer);
		for (int node = 1; node <= MultiLayer.m_NumNodes[layer]; ++node) {
			printf("Node %d: %lf\n", node, MultiLayer.m_ErrorGradient[layer][node]);
		}
	}

	printf("\nWeights after learning:\n");
	for (int layer = 0; layer < MultiLayer.m_iNumTotalLayer - 1; ++layer) {
		printf("Layer %d to Layer %d:\n", layer, layer + 1);
		for (int snode = 0; snode <= MultiLayer.m_NumNodes[layer]; ++snode) {
			for (int enode = 1; enode <= MultiLayer.m_NumNodes[layer + 1]; ++enode) {
				printf("Weight[%d][%d][%d]: %lf\n", layer, snode, enode, MultiLayer.m_Weight[layer][snode][enode]);
			}
		}
	}

	return 0;
}


/*
int main() {
	int epoch,n = 0;
	int input[4][2] = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	int danswer[4] = { 0,1,1,0 };
	int HiddenNodes[2] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(2, HiddenNodes, 1, numofHiddenLayer);
	// 입력과 현재의 출력값 계산
	for (n = 0; n < 4; n++)
	{
		MultiLayer.pInValue[1] = input[n][0];
		MultiLayer.pInValue[2] = input[n][1];
		MultiLayer.Forward();
		printf("%lf %lf =%lf\n", MultiLayer.pInValue[1], MultiLayer.pInValue[2], MultiLayer.pOutValue[1]);
	}
	printf("\n");
	getchar();

	double MSE;
	printf("******** 학습시작 ****************\n");
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		for (n = 0; n < 4; n++)
		{
			MultiLayer.pInValue[1] = input[n][0];	// 입력전달
			MultiLayer.pInValue[2] = input[n][1];	// 입력전달
			MultiLayer.pCorrectOutValue[1] = danswer[n];	// 정답전달

			MultiLayer.Forward();					// 출력계산
			MultiLayer.BackPropagationLearning();	// 학습-가중치갱신

			// 갱신후에 에러값을 계산
			MultiLayer.Forward();					// 갱신이후의 출력 계산
			MSE += (MultiLayer.pCorrectOutValue[1] - MultiLayer.pOutValue[1]) *
				(MultiLayer.pCorrectOutValue[1] - MultiLayer.pOutValue[1]);
		}
		MSE /= 4;	// 평균값 계산
		printf("Epoch%d(MSE)=%lf\n", epoch, MSE);
		if (MSE < 0.0001)
			break;
	}
	printf("******** 학습종료 ****************\n");

	// 입력과 현재의 출력값 계산
	for (n = 0; n < 4; n++)
	{
		MultiLayer.pInValue[1] = input[n][0];
		MultiLayer.pInValue[2] = input[n][1];
		MultiLayer.Forward();
		printf("%lf %lf =%lf\n", MultiLayer.pInValue[1], MultiLayer.pInValue[2], MultiLayer.pOutValue[1]);
	}
	printf("\n");

}
*/