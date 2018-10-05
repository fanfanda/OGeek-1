#OGeek中Word2Vec Lzz&Xlxw
#Ver In 2018.10.5
#eg.本代码对我们自己的语料库完成了词->向量的转化，解决了原来语料的缺失部分问题
#Update
#Finish Fundamental Code [10.5]
#-----------------------------------------------------------------------------
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba.analyse
import codecs
import numpy as np

#对原始预料分词
def WordSplit(Input_Path,Output_path,Process = False):
    '''
    : Args   :Input_Path   : 输入txt文件的地址
    : Args   :Output_path  : 输出分词后txt文件的地址
    '''
    #判断有无必要进行处理
    if Process == False:
        return
    #以写的方式打开原始的简体中文语料库
    InTxt = codecs.open(Input_Path,'r',encoding="utf8")
    #将分完词的语料写入到输出文件中
    TargetTxt = codecs.open(Output_path, 'w',encoding="utf8")
    print('Start Split Words.Waiting..')
    #处理数据
    line_num = 0
    for line in InTxt:
        line_seg = " ".join(jieba.cut(line))
        TargetTxt.writelines(line_seg)
        line_num = line_num + 1
    print('Split Finished,Total Preocess {} Lines'.format(line_num))
    #关闭文件
    InTxt.close()
    TargetTxt.close()
    
#gensim训练词向量
def VecterTrain(Dimesion,Window,MinFreq,InTxtPath,ModelPath,VecterPath):
    '''
    : Args   :Dimesion     : 词向量维度(d)
    : Args   :Window       : 扫描窗口大小
    : Args   :MinFreq      : 最低词频限制
    : Args   :InTxtPath    : 分词后的txt地址
    : Args   :ModelPath    : 模型保存地址
    : Args   :VecterPath   : 词向量保存地址
    '''
    print('Training Word2Vec.Waiting..')
    print('Use Max CPU:Total 【{}】'.format(multiprocessing.cpu_count()))
    model = Word2Vec(LineSentence(InTxtPath), size = Dimesion, window = Window, min_count = MinFreq, workers=multiprocessing.cpu_count())
    print('Training Over..Saving Model and Vecter.Waiting..')
    #模型存储
    model.save(ModelPath)
    #词向量存储
    model.wv.save_word2vec_format(VecterPath, binary=False)

def main():
    #对语料文件进行分词
    WordSplit('Data_In.txt','Data_Split.txt',True)
    #对数据进行词向量训练
    VecterTrain(256,10,1,'Data_Split.txt','OGeek_WordModel.model','OGeek_WordVecter.vecter')

main()