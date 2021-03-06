from pyhanlp import JClass
from pyhanlp import SafeJClass
import os

# 当前文件路径
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(FILE_PATH, 'train_data.txt')

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
def divisionTrainData(filepath):
    with open(filepath, 'r', encoding='utf-8') as fin:
        print('成功加载训练集。')



divisionTrainData(TRAIN_DATA_PATH)
