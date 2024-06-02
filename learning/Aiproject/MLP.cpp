
#include "MLP.h"

#include <malloc.h> //malloc()�� ���� �߰�
#include <stdlib.h> //srand()�������߰�
#include <stdio.h>
#include <time.h> // time()������ �߰�
#include <math.h> // exp()������ �߰�

CMLP::CMLP()
{   
    int layer=0;
    m_iNumInNodes = 0;
    m_iNumOutNodes = 0;
    m_NumNodes = NULL;
    m_iNumHiddenLayer = 0;
    m_iNumTotalLayer = 0;
    m_Weight = NULL;
    m_NodeOut = NULL;
    pInValue = NULL;
    pOutValue = NULL;	
    pCorrectOutValue = NULL;
    m_iNumInNodes = 0;
    m_iNumOutNodes = 0;
    m_NumNodes = NULL;
    m_ErrorGradient = NULL;

}

CMLP::~CMLP()
{
    int layer, snode;
    if (m_NodeOut != NULL)
    {
        for (layer = 0; layer < (m_iNumTotalLayer + 1); layer++)
            free(m_NodeOut[layer]);
        free(m_NodeOut);
    }

    if (m_Weight != NULL)
    {
        for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++)
        {
            if (m_Weight[layer] != NULL)
            {
                for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
                    free(m_Weight[layer][snode]);
                free(m_Weight[layer]);
            }
        }
    }

    if (m_ErrorGradient != NULL)
    {
        for (layer = 0; layer < (m_iNumTotalLayer); layer++)
            free(m_ErrorGradient[layer]);
        free(m_ErrorGradient);
    }
}



bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer)
{
    int layer, snode, enode;

    m_iNumInNodes = InNode;
    m_iNumOutNodes = OutNode;
    m_iNumHiddenLayer = numHiddenLayer;                // �Է�,����� ����
    m_iNumTotalLayer = numHiddenLayer + 2;	// ����+�Է�+���

    //m_iNumNodes�� ���� �޸��Ҵ�
    m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));    // ����(+1)

    m_NumNodes[0] = m_iNumInNodes;
    for (layer = 0; layer < m_iNumHiddenLayer; layer++)
        m_NumNodes[1 + layer] = pHiddenNode[layer];
    m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;    // ����� ����
    m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;         // ����   ����
    // ����庰 ��¸޸��Ҵ�=[layerno][nodeno]
    // �Է�:m_NodeOut[0][],���m_NodeOut[m_iNumTotalLayer-1][]
    // ����:m_NodeOut[m_iNumTotalLayer][]
    m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));              // ����(+1)
    for (layer = 0; layer < m_iNumTotalLayer; layer++)
        m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));    // ���̾�� ���� +1
    // ����(��� ���� ���� ����,���̾�� �ʿ������ ÷�ڴ� 1���� n����)
    m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer - 1] + 1) * sizeof(double));

    // ����ġ �޸��Ҵ� m_Weight[����layer][���۳��][������]
    m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
    for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
    {
        m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));       // ���̾(+1)
        for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
            m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));	// �������̾��� ����
    }

    pInValue = m_NodeOut[0];
    pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
    pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

    InitW();

    // ���̾�� ���� ��°�=1
    for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
    {
        m_NodeOut[layer][0] = 1;
    }
    return true;
}

double CMLP::ActivationFunc(double weightsum)
{
    // step func
    //if (weightsum > 0)	return 1.0;
    //else	return 0.0;

    // sigmoid func
   return 1.0 / (1.0 + exp(-weightsum));
}

void CMLP::InitW()
{
    int layer, snode, enode;

    srand(time(NULL));
    for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
    {
        for (snode = 0; snode <= m_NumNodes[layer]; snode++) 	// for ���̾�� ���� 0����
        {
            for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) // ���� ���̾��� ����
            {
                m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;     // -0.5~0.5
            }
        }
    }
}

void CMLP::BackPropagationLearning()
{
    int layer;
    // ������縦 ���� �޸� �Ҵ�
    if (m_ErrorGradient == NULL)
    {
        // ����庰 ��¸޸��Ҵ�=m_ErrorGrident[layerno][nodeno]
        // �Է�:m_ErrorGradient[0][],���m_ErrorGradient[m_iNumTotalLayer-1][]
        m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));	//
        for (layer = 0; layer < m_iNumTotalLayer; layer++)
            m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));		// ���̾(0)�� ���� +1
    }

    int snode, enode, node;
    // �����error�����
    for (node = 1; node <= m_iNumOutNodes; node++)
    {
        m_ErrorGradient[m_iNumTotalLayer - 1][node] =
            (pCorrectOutValue[node] - m_NodeOut[m_iNumTotalLayer - 1][node])
            * m_NodeOut[m_iNumTotalLayer - 1][node] * (1 - m_NodeOut[m_iNumTotalLayer - 1][node]);
    }

    // error�����
    for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
    {
        for (snode = 1; snode <= m_NumNodes[layer]; snode++)
        {
            m_ErrorGradient[layer][snode] = 0.0;
            for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
            {
                m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
            }
            m_ErrorGradient[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
        }
    }

    // ����ġ����
    for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
    {
        for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
        {
            m_Weight[layer][0][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * 1);// ���̾
            for (snode = 1; snode <= m_NumNodes[layer]; snode++)
            {
                m_Weight[layer][snode][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * m_NodeOut[layer][snode]);
            }
        }
    }
}


void CMLP::Forward()
{
    int layer, snode, enode;
    double wsum; // ������

    for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
    {
        for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
        {
            wsum = 0.0;	// ��庰 ������
            wsum += m_Weight[layer][0][enode] * 1;	//���̾ 
            for (snode = 1; snode <= m_NumNodes[layer]; snode++)
            {
                wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
            }

            m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
        }
    }
}


bool CMLP::SaveWeight(char* fname)
{
    // TODO: ���⿡ ���� �ڵ� �߰�.
    int layer, snode, enode;
    FILE* fp;

    if ((fp = fopen(fname, "wt")) == NULL)
        return false;

    // �Է³��� ���緹�̾�� ��³���
    fprintf(fp, "%d %d %d\n", m_iNumInNodes, m_iNumHiddenLayer, m_iNumOutNodes);
    // node_layer0 node_layer1 node_layer2......
    for (layer = 0; layer < m_iNumTotalLayer; layer++)
    {
        fprintf(fp, "%d ", m_NumNodes[layer]);
    }
    fprintf(fp, "\n");

    // save weight
    for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
    {
        for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
        {
            for (snode = 0; snode <= m_NumNodes[layer]; snode++)// ���̾������ 0����
            {
                fprintf(fp, "%.9lf ", m_Weight[layer][snode][enode]);
            }
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return true;
}


bool CMLP::LoadWeight(char* fname)
{
    // TODO: ���⿡ ���� �ڵ� �߰�.
    int layer, snode, enode;
    FILE* fp;

    if ((fp = fopen(fname, "rt")) == NULL)
        return false;

    // �Է³��� ���緹�̾�� ��³���
    fscanf(fp, "%d %d %d", &m_iNumInNodes, &m_iNumHiddenLayer, &m_iNumOutNodes);
    // node_layer0 node_layer1 node_layer2......
    for (layer = 0; layer < m_iNumTotalLayer; layer++)
    {
        fscanf(fp, "%d ", &m_NumNodes[layer]);
    }

    // load weight
    for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
    {
        for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
        {
            for (snode = 0; snode <= m_NumNodes[layer]; snode++)// ���̾������ 0����
            {
                fscanf(fp, "%lf ", &m_Weight[layer][snode][enode]);
            }
        }
    }
    fclose(fp);

    return true;
}
