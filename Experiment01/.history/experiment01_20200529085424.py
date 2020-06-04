from pyhanlp import JClass
from pyhanlp import SafeJClass
import os
from pyhanlp.static import download

# 各类文件路径
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, 'train_data.txt')
CLASSIFICATION_DATA_PATH = os.path.join(PROJECT_PATH, 'DataSet')
SAVE_MODEL_PATH = PROJECT_PATH + '/negetive.model'
###############################################################################
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
所以我们要先将训练数据拆分成满足Hanlp的目录结构文件
'''


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
MemoryDataSet = JClass('com.hankcs.hanlp.classification.corpus.MemoryDataSet')
Evaluator = JClass(
    'com.hankcs.hanlp.classification.statistics.evaluations.Evaluator')


##########################################################################################
# 对数据集进行预处理
def dataPreprocessing(dataPath):
    print("开始修正数据......")
    fi = open(dataPath, "r", encoding="utf-8")
    fi.readline()
    dataPath = PROJECT_PATH + "/train_data(preprocess).txt"
    if (os.path.isfile(dataPath)):
        print('数据之前已被修正......')
        return
    fo = open(dataPath, "a+", encoding="utf-8")
    fo.write('qid\ttext\tlabel\n')
    negetive_list = [
        'nm', 'tm', 'fw', 'laji', 'sb', 'zz', '低能', '全家', '爸', '尼玛', '鸡', '干嘛',
        '曰', '草', '艹', '傻', 'sha', 'Sha', '演', '挂', '送', '投', '颂', '狗', '妈',
        'ma', '臭b', '马', '你妹', 'jb', 'song', '宋'
    ]
    positive_list = [
        '皮肤', '别挂', '别送', '别投', '别演', '不要送', '不要挂', '不要投', '不要演', '表演', '举报'
    ]
    modifiDataNum = 0
    for sentence in fi.readlines():
        sentence = sentence.strip('\n')  # 去除末尾的\n
        sentence = sentence.split('\t')  # 去除空格
        prelabel = sentence[2]
        for negetiveText in negetive_list:
            if (negetiveText in sentence[1]):
                sentence[2] = '1'
        for positiveText in positive_list:
            if (positiveText in sentence[1]):
                sentence[2] = '0'
        if (not sentence[2] == prelabel):
            modifiDataNum += 1
        fo.write(sentence[0] + '\t' + sentence[1] + '\t' + sentence[2] + '\n')
    print("共计修正数据：", modifiDataNum, "条")
    fi.close()
    fo.close()
    return dataPath


##########################################################################################
# 评分
def valuation(dataPath, range, classifier):
    TP, TN, FP, FN = 0
    fi = open(dataPath, 'r', encoding='utf-8')
    fi.readline()
    textlist = fi.readlines()
    if (range < 0 and range >= -1):
        for text in textlist[range * len(textlist):]:
            classifier
    elif (range >= 0 and range <= 1):
        for text in textlist[:range * len(textlist)]:
            pass
    else:
        print("range value is false")
        fi.close()
        return

    fi.close()


##########################################################################################

if __name__ == '__main__':
    dataPath = dataPreprocessing(TRAIN_DATA_PATH)
    divisionTrainData(dataPath, CLASSIFICATION_DATA_PATH)
    # 使用前90%的数据进行训练，使用后10%数据进行验证
    training_corpus = FileDataSet().setTokenizer(BigramTokenizer()).load(
        CLASSIFICATION_DATA_PATH, "UTF-8", 0.9)
    classifier = LinearSVMClassifier()
    classifier.train(training_corpus)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, SAVE_MODEL_PATH)
    # 使用后10%的数据进行验证
    valuation(dataPath, -0.1, classifier)
