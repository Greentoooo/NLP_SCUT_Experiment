from pyhanlp import JClass
from pyhanlp import SafeJClass
import os

# 当前文件路径
FILEPATH = os.path.dirname(os.path.realpath(__file__))

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
    f = open(FILEPATH,'r')
    pass
