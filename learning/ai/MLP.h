#define LEARNING_RATE	0.1

#define MAX_EPOCH 10000000


class CMLP {
public:
    CMLP();   // 생성자
    ~CMLP();  // 소멸자
    bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
    bool SaveWeight(char* fname);
    bool LoadWeight(char* fname);
    void Forward();
    void BackPropagationLearning();
    int m_iNumInNodes;
    int m_iNumOutNodes;
    int m_iNumHiddenLayer;  // 히든레이어의수 (hidden only)
    int m_iNumTotalLayer;     // 전체레이어의 수 (inputlayer+hiddenlayer+outputlayer)
    int* m_NumNodes;
    double*** m_Weight;	// [시작layer][시작노드][연결노드]
    double** m_NodeOut;	// [layer][node]
    double* pInValue, * pOutValue;		// 입력레이어,출력레이어
    double* pCorrectOutValue;		// 정답레이어
    double** m_ErrorGradient;	//[layer][node]



private:
    double ActivationFunc(double weightsum);
    void InitW();
};

