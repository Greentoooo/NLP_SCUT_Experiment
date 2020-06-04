from pyhanlp import JClass
from pyhanlp import SafeJClass
import os

# 当前文件路径
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, 'train_data.txt')
CLASSIFICATION_DATA_PATH = os.path.join(PROJECT_PATH, 'DataSet')
'''
Hanlp的文本分类将每个文本视作一个文档，类目和文档满足如下目录结构：
根目录
├── 分类A
│   └── 1.txt
│   └── 2.txt
│   └── 3.txt
├── 分类B
│   └── 1.txt
│   └── ...
└── ...
'''


# 将训练数据拆分成满足Hanlp的目录结构文件
def divisionTrainData(traimDataPath, projectPath):
    # 创建类别目录
    positivePath = os.path.join(projectPath, 'positive')
    negetivePath = os.path.join(projectPath, 'negetive')
    os.mkdir(projectPath)
    os.mkdir(positivePath)
    os.mkdir(negetivePath)

    with open(filePath, 'r', encoding='utf-8') as fin:
        print('成功加载训练集。')
        fin.readline()
        print(fin.readline)


divisionTrainData(TRAIN_DATA_PATH, POSITIVE_DATA_PATH, NEGETIVE_DATA_PATH)
