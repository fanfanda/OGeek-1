#OGeek中句子与向量转化 Lzz&Xlxw
#Ver In 2018.10.1
#eg.本代码作为示例对前缀进行了转化
#使用的语料以及词嵌入向量为:sjl_weixin|Author:苏剑林|Skip-Gram, Huffman Softmax, 窗口大小 10, 最小词频 64, 迭代 10 次
#-----------------------------------------------------------------------------

#Import Lib
import numpy as np
import jieba
import bcolz
import json
import scipy.io as sio
import scipy.io as scio
import time

#词嵌入向量读取
def load_embeddings(folder_path):
    """从 bcolz 加载 词/字 向量

    Args:
        - folder_path (str): 解压后的 bcolz rootdir（如 zh.64），
                             里面包含 2 个子目录 embeddings 和 words，
                             分别存储 嵌入向量 和 词（字）典

    Returns:
        - words (bcolz.carray): 词（字）典列表（bcolz carray  具有和 numpy array 类似的接口）
        - embeddings (bcolz.carray): 嵌入矩阵，每 1 行为 1 个 词向量/字向量，
                                     其行号即为该 词（字） 在 words 中的索引编号
    """
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words'%folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%folder_path, mode='r')
    return words, embeddings

#输入已存的数据
def input_data(MatFile_path,JsonFile_path):
    MatFile = scio.loadmat(MatFile_path)
    prefix = MatFile['prefix']
    title = MatFile['title']
    tag = MatFile['tag']
    label = MatFile['label']

    JsonFile = open(JsonFile_path,encoding='utf-8')
    prediction = json.load(JsonFile)
    return prefix, prediction, title, tag, label

#分词后导出平均Prefix句向量
def Prefix2AVector(Prefix,Words,Embeddings):
    ReturnArray = np.zeros([1,256])
    TimeIndex   = 0
    Total       = len(Prefix)
    timeCount   = time.time()
    for i in range(0,len(Prefix)):
        TimeIndex = TimeIndex + 1
        if TimeIndex % 500 == 0:
            print('500 Prefix has used 【{:.2f}】s,Rest 【{:.2f}】% Prefix'.format((time.time() - timeCount),100 - TimeIndex/Total))
            timeCount   = time.time()
        np.append(ReturnArray,PrefixConverter(Prefix[i],Words,Embeddings),axis = 0)
    return ReturnArray[1:]


#句子 -> 向量转化    
def PrefixConverter(Sentence,Words,Embeddings):
    ReturnArray = np.zeros([1,256])
    TmpSentence = Sentence.replace(' ','')
    Cutls = jieba.lcut(TmpSentence)
    TmpCount = 0
    #得到所有分词结果的词向量
    for i in Cutls:
        index = np.where(Words == i)[0]
        if len(index) == 0:
            continue
        else:
            TmpCount = TmpCount + 1
            ReturnArray = Embeddings[index] + ReturnArray
    #得到平均词向量均值
    return ReturnArray / TmpCount

#保存句向量
def Vecter2mat(Vecter,path):
    sio.savemat(path+'PrefixVec.mat',{'PrefixVec':Vecter})

def main():
    Words,Embeddings = load_embeddings('/Users/xulvxiaowei/Downloads/sjl_weixin/zh.256')
    Prefix, Prediction, Title, Tag, Label = input_data('OGeekData.mat','Ogeek.json')
    PrefixVecter = Prefix2AVector(Prefix,np.array(Words),np.array(Embeddings))
    Vecter2mat(PrefixVecter,'/')

main()
