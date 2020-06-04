from pyhanlp import JClass
from pyhanlp import SafeJClass
import os
import shutil

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
├── 分类B
│   └── 1.txt
│   └── ...
└── ...
'''


###############################################################################
# 将训练数据拆分成满足Hanlp的目录结构文件
def divisionTrainData(trainDataPath, classificationPath):
    # 创建类别目录
    positivePath = os.path.join(classificationPath, 'positive')
    negetivePath = os.path.join(classificationPath, 'negetive')
    if not os.path.isdir(classificationPath):
        os.mkdir(classificationPath)
    if not os.path.isdir(positivePath):
        os.mkdir(positivePath)
    if not os.path.isdir(negetivePath):
        os.mkdir(negetivePath)

    # 将文本内容按照label分成积极和消极两个类目并保存在不同文件夹
    print('开始分类训练集..........')
    with open(trainDataPath, 'r', encoding='utf-8') as fin:
        fin.readline()
        for sentence in fin.readlines():
            sentence = sentence.strip('\n')  # 去除末尾的\n
            sentence = sentence.split('\t')  # 去除空格
            if (sentence[2] == '0'):
                pf = open(os.path.join(positivePath, sentence[0] + '.txt'),
                          'a+',
                          encoding='utf-8')
                pf.write(sentence[1])
                pf.close()
            else:
                nf = open(os.path.join(negetivePath, sentence[0] + '.txt'),
                          'a+',
                          encoding='utf-8')
                nf.write(sentence[1])
                nf.close()
    fin.close()
    print('成功分类训练集..........')


##########################################################################################
# 退出程序前删除分类的训练集
def endDeleteDataSet(dataSetPath):
    if os.path.isdir(dataSetPath):
        shutil.rmtree(dataSetPath)
        print('已删除分类后的训练集。')
    else:
        print('训练集未分类。')


##########################################################################################

# 载入分类器
IClassifier = JClass('com.hankcs.hanlp.classification.classifiers.IClassifier')
NaiveBayesClassifier = JClass(
    'com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
LinearSVMClassifier = SafeJClass(
    'com.hankcs.hanlp.classification.classifiers.LinearSVMClassifier')

# 载入分词器
ITokenizer = JClass('com.hankcs.hanlp.classification.tokenizers.ITokenizer')
HanLPTokenizer = JClass(
    'com.hankcs.hanlp.classification.tokenizers.HanLPTokenizer')
BigramTokenizer = JClass(
    'com.hankcs.hanlp.classification.tokenizers.BigramTokenizer')

##########################################################################################

if __name__ == '__main__':
    divisionTrainData(TRAIN_DATA_PATH, CLASSIFICATION_DATA_PATH)
    classifier = LinearSVMClassifier()
    classifier.train(CLASSIFICATION_DATA_PATH)
    print(classifier.classify("我去挂机了"))
    endDeleteDataSet(CLASSIFICATION_DATA_PATH)
