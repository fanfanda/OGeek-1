#OGeek中句子与向量转化 Lzz&Xlxw
#Ver In 2018.10.1
#eg.本代码对前面解析得到的数据进行了词向量的转化
#使用的语料以及词嵌入向量为:sjl_weixin|Author:苏剑林|Skip-Gram, Huffman Softmax, 窗口大小 10, 最小词频 64, 迭代 10 次
#Update
#1.Bug Fix 2018.10.2
#2.Bug Fix 2018.10.3
#-----------------------------------------------------------------------------

#Import Lib
import numpy as np
import jieba
import bcolz
import json
import scipy.io as sio
import scipy.io as scio
import time

#全局设定参数
Params = [256,500]
#Params = [词嵌入维度,每一次读取多少数据提示一次]

#从 bcolz 加载 词/字 向量
def load_embeddings(folder_path):
    '''
    ：older_path (str): 解压后的 bcolz rootdir（如 zh.64），
                               里面包含 2 个子目录 embeddings 和 words，
                               分别存储 嵌入向量 和 词（字）典
    '''
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words'%folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%folder_path, mode='r')
    return words, embeddings

#输入已存的数据
def input_data(MatFile_path,JsonFile_path):
    '''
    MatFile_path  : .Mat文件存放地址
    JsonFile_path : .json文件存放地址
    '''
    MatFile = scio.loadmat(MatFile_path)
    prefix = MatFile['prefix']
    title = MatFile['title']
    tag = MatFile['tag']
    label = MatFile['label']

    JsonFile = open(JsonFile_path,encoding='utf-8')
    prediction = json.load(JsonFile)
    return prefix, prediction, title, tag, label

#JSON分词后导出平均语句向量
def Prediction2AVector(Prediction,Words,Embeddings):
    '''
    :Prediction   : 待转换为向量的JSon组
    :Words        : 语料中的词汇组
    :Embeddings   : 词嵌入向量组
    '''
    prediction_vector = np.zeros([len(Prediction)*10,Params[0]])
    timeCount   = time.time()
    Total       = len(Prediction)
    #punctuation=['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'"]
    q = 0
    for keys in Prediction:
        j = 0
        for key in Prediction[keys]:
            TmpCount = 0
            vector = np.zeros([1,Params[0]])
            predict_list = jieba.lcut(key,cut_all=False)
            #predict_list = [x for x in predict_list if x not in punctuation]
            for i in range(len(predict_list)):
                itemindex = np.where(Words == predict_list[i])
                if len(itemindex[0]) == 0:
                    continue
                else:
                    TmpCount+=1
                    vector += Embeddings[itemindex][:]
            prediction_vector[q*10+j][:] = vector/TmpCount
            j+=1
        q+=1
        if q % Params[1] == 0:
            print('{} {}s has used 【{:.2f}】s,Rest 【{:.2f}】% {}'.format(Params[1],'Prediction',(time.time() - timeCount),100 - q/Total*100,'Predictions'))
            timeCount   = time.time()
    return prediction_vector

#分词后导出平均句向量
def Element2AVector(Element,Words,Embeddings,LeadStr='Element'):
    '''
    :Element    : 需要转换为向量的句子组 eg.Prefix|Title etc.
    :Words      : 语料中的词汇组
    :Embeddings : 词嵌入向量组
    :LeadStr    : 作为提示(可缺省)
    '''
    ReturnArray = np.zeros([len(Element),Params[0]])
    TimeIndex   = 0
    Total       = len(Element)
    timeCount   = time.time()
    for i in range(0,len(Element)):
        ReturnArray[TimeIndex] = SentenceConverter(Element[i],Words,Embeddings)
        TimeIndex = TimeIndex + 1
        if TimeIndex % Params[1] == 0:
            print('{} {}s has used 【{:.2f}】s,Rest 【{:.2f}】% {}'.format(Params[1],LeadStr,(time.time() - timeCount),100 - TimeIndex/Total*100,LeadStr))
            timeCount   = time.time()
    return ReturnArray

#句子 -> 向量转化    
def SentenceConverter(Sentence,Words,Embeddings):
    '''
    :Sentence   : 待转换为向量的句子
    :Words      : 语料中的词汇组
    :Embeddings : 词嵌入向量组
    '''
    ReturnArray = np.zeros([1,Params[0]])
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
    return ReturnArray / TmpCount  #NaN用于判定数据无效
    
#保存句向量
def Vecter2mat(Vecter,path,Tag):
    '''
    :Vecter : 输入需要保存的向量
    :Path   : 输入保存的路径 
    :Tag    : 输入保存进入Mat的内置变量名称 
    '''
    sio.savemat(path+'.mat',{'PrefixVec':Vecter})

def main():
    Words,Embeddings = load_embeddings('/Users/xulvxiaowei/Downloads/sjl_weixin/zh.256')
    Prefix, Prediction, Title, Tag, Label = input_data('/Users/xulvxiaowei/Documents/GitHub/OGeek/OGeekDataParser/OGeekData.mat','/Users/xulvxiaowei/Documents/GitHub/OGeek/OGeekDataParser/Ogeek.json')
    Words = np.array(Words)
    Embeddings = np.array(Embeddings)
    '''
    #前缀转换为词嵌入
    PrefixVecter = Element2AVector(Prefix,Words,Embeddings,'Prefix')
    Vecter2mat(PrefixVecter,'PrefixVec','PrefixVec')
    #标题转换为词嵌入
    TitleVecter = Element2AVector(Title,Words,Embeddings,'Title')
    Vecter2mat(TitleVecter,'TitlefixVec','TitlefixVec')
    #类别转换为词嵌入
    TagVecter = Element2AVector(Tag,Words,Embeddings,'Tag')
    Vecter2mat(TagVecter,'TagVec','TagVec')
    '''
    #预测转换为词嵌入
    PredictVec = Prediction2AVector(Prediction,Words,Embeddings)
    Vecter2mat(PredictVec,'RredictVec','RredictVec')
main()
