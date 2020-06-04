#coding=utf-8

#################################
# 情感分析之消极言语识别，主程序模板
# file: main_test.py
#################################

import os
from optparse import OptionParser
from pyhanlp import SafeJClass
from pyhanlp.static import STATIC_ROOT, download


###################################
# arg_parser： 读取参数列表
###################################
def arg_parser():
    oparser = OptionParser()

    oparser.add_option("-m",
                       "--model_file",
                       dest="model_file",
                       help="输入模型文件 \
            must be: negative.model",
                       default=None)

    oparser.add_option("-d",
                       "--data_file",
                       dest="data_file",
                       help="输入验证集文件 \
            must be: validation_data.txt",
                       default=None)

    oparser.add_option("-o",
                       "--out_put",
                       dest="out_put_file",
                       help="输出结果文件 \
			must be: result.txt",
                       default=None)

    (options, args) = oparser.parse_args()
    global g_MODEL_FILE
    g_MODEL_FILE = str(options.model_file)

    global g_DATA_FILE
    g_DATA_FILE = str(options.data_file)

    global g_OUT_PUT_FILE
    g_OUT_PUT_FILE = str(options.out_put_file)


###################################
# load_model： 加载模型文件
###################################
def load_model(model_file_name):
    def install_jar(name, url):
        dst = os.path.join(STATIC_ROOT, name)
        if os.path.isfile(dst):
            return dst
        download(url, dst)
        return dst

    install_jar(
        'text-classification-svm-1.0.2.jar',
        'http://file.hankcs.com/bin/text-classification-svm-1.0.2.jar')
    install_jar('liblinear-1.95.jar',
                'http://file.hankcs.com/bin/liblinear-1.95.jar')

    # 载入分类器
    LinearSVMClassifier = SafeJClass(
        'com.hankcs.hanlp.classification.classifiers.LinearSVMClassifier')
    # 保存和解析模型的工具
    IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
    return LinearSVMClassifier(IOUtil.readObjectFrom(model_file_name))


###################################
# predict： 根据模型预测结果并输出结果文件，文件内容格式为qid\t言语\t标签
###################################
def predict(model):
    print("predict start.......")
    ###################################
    # 预测逻辑和结果输出，("%d\t%s\t%d", qid, content, predict_label)
    ###################################
    fin=open

    print("predict end.......")

    return None


###################################
# main： 主逻辑
###################################
def main():
    print("main start.....")

    if g_MODEL_FILE is not None:
        model = load_model(g_MODEL_FILE)
        predict(model)

    print("main end.....")

    return 0


if __name__ == '__main__':
    arg_parser()
    main()
