#define LEARNING_RATE	0.1

#define MAX_EPOCH 10000000


class CMLP {
public:
    CMLP();   // ������
    ~CMLP();  // �Ҹ���
    bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
    bool SaveWeight(char* fname);
    bool LoadWeight(char* fname);
    void Forward();
    void BackPropagationLearning();
    int m_iNumInNodes;
    int m_iNumOutNodes;
    int m_iNumHiddenLayer;  // ���緹�̾��Ǽ� (hidden only)
    int m_iNumTotalLayer;     // ��ü���̾��� �� (inputlayer+hiddenlayer+outputlayer)
    int* m_NumNodes;
    double*** m_Weight;	// [����layer][���۳��][������]
    double** m_NodeOut;	// [layer][node]
    double* pInValue, * pOutValue;		// �Է·��̾�,��·��̾�
    double* pCorrectOutValue;		// ���䷹�̾�
    double** m_ErrorGradient;	//[layer][node]



private:
    double ActivationFunc(double weightsum);
    void InitW();
};

