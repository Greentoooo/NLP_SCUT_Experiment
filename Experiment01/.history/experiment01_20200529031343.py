from pyhanlp import JClass
from pyhanlp import SafeJClass
import os
from pyhanlp.static import download

# 各类文件路径
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, 'train_data.txt')
CLASSIFICATION_DATA_PATH = os.path.join(PROJECT_PATH, 'DataSet')
SAVE_MODEL_PATH = PROJECT_PATH + '/negetive.model'
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
    # 若存在DataSet文件夹，说明先前已经对数据集进行过分类，直接return
    if not os.path.isdir(classificationPath):
        os.mkdir(classificationPath)
    else:
        return
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
# 下载基于支持向量机的文本分类器的第三方依赖
def install_jar(name, filepath, url):
    dst = os.path.join(filepath, name)
    if os.path.isfile(dst):
        return dst
    download(url, dst)
    return dst


install_jar('text-classification-svm-1.0.2.jar', PROJECT_PATH,
            'http://file.hankcs.com/bin/text-classification-svm-1.0.2.jar')
install_jar('liblinear-1.95.jar', PROJECT_PATH,
            'http://file.hankcs.com/bin/liblinear-1.95.jar')
##########################################################################################

# 载入分词器
BigramTokenizer = JClass(
    'com.hankcs.hanlp.classification.tokenizers.BigramTokenizer')

# 载入分类器
LinearSVMClassifier = SafeJClass(
    'com.hankcs.hanlp.classification.classifiers.LinearSVMClassifier')

# 保存模型的工具
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')

FileDataSet = JClass('com.hankcs.hanlp.classification.corpus.FileDataSet')


# 载入模型或新建模型
def train_or_load_classifier():
    model_path = SAVE_MODEL_PATH
    if os.path.isfile(model_path):
        return LinearSVMClassifier(IOUtil.readObjectFrom(model_path))
    classifier = LinearSVMClassifier()
    classifier.train(CLASSIFICATION_DATA_PATH)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return LinearSVMClassifier(model)


##########################################################################################
# 对数据集进行预处理
def dataPreprocessing():
    pass


##########################################################################################

##########################################################################################

if __name__ == '__main__':
    divisionTrainData(TRAIN_DATA_PATH, CLASSIFICATION_DATA_PATH)
    training_corpus = FileDataSet().setTokenizer(LinearSVMClassifier()).load(TRAIN_DATA_PATH, "UTF-8", 0.9)

    print(classifier.classify("我去挂机了"))
