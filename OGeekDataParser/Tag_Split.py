#OGeek的Mini训练集制作 Lzz&Xlxw
#Ver In 2018.10.3
#-----------------------------------------------------------------------------

#import Lib
import scipy.io as sio
import scipy.io as scio
import json
import numpy as np

#全局参数:
Params  = [10000] # 需要的训练集大小
n_Prefix,n_Title,n_Tag,n_Label = [],[],[],[]
n_Prediction = {}


#输入已存的数据
def input_data(MatFile_path,JsonFile_path):
    '''
    :MatFile_path  : .Mat文件存放地址
    :JsonFile_path : .json文件存放地址
    '''
    MatFile = scio.loadmat(MatFile_path)
    prefix = MatFile['prefix']
    title = MatFile['title']
    tag = MatFile['tag']
    label = MatFile['label']

    JsonFile = open(JsonFile_path,encoding='utf-8')
    prediction = json.load(JsonFile)
    return prefix, prediction, title, tag, label

#读取Tag得到Traning Mini Batch的数量
def TagSplit(Train_Tags,Train_Labels,Prefix,Prediction,Title):
    '''
    :Train_Tags   : 训练集_Tag属性
    :Train_Labels : 训练集_标签
    '''
    #Step1:Tags频次计数
    Tags      = {}
    PosTags   = {}
    NegTags   = {}
    DealIndex = 0

    for i in Train_Tags:
        if i not in Tags:
            Tags[i]    = 1
            PosTags[i] = 0
            NegTags[i] = 0
        else:
            Tags[i] += 1
        #记录正负样本
        if int(Train_Labels[DealIndex]) == 1:
            PosTags[i] += 1
        else:
            NegTags[i] += 1
        DealIndex += 1
    #Step 2:
    GetLs = np.zeros([len(Tags),2])
    TmpIndex = 0
    OrderLs = []
    #运算得到Tags分别子集数量
    for j in Tags:
        OrderLs.append(j)
        GetLs[TmpIndex,0] = round((Tags[j] / len(Train_Tags) * Params[0] * PosTags[j]/(PosTags[j]+NegTags[j])))
        GetLs[TmpIndex,1] = round((Tags[j] / len(Train_Tags) * Params[0] * NegTags[j]/(PosTags[j]+NegTags[j])))
        print(i,GetLs[TmpIndex])
    #补救
    GetLs[-1,0] = Params[0] - (sum(GetLs[:,0]) - GetLs[-1,0])
    GetLs[-1,1] = Params[0] - (sum(GetLs[:,1]) - GetLs[-1,1])
    
    '''
    #补救GetLs里面的空值0
    TmpPosAdd = np.where(GetLs[:,0] == 0)[0]
    TmpNegAdd = np.where(GetLs[:,1] == 0)[0]
    '''
    print(OrderLs)
    #Step 3:
    TmpIndex = 0
    RunSym1   = False
    RunSym2   = False
    Count     = 0
    #得到最终的分配方式
    for k in Train_Tags:
        if RunSym1 == True and RunSym2 == True:
            return
        if Train_Labels[TmpIndex]==1:
            #Pos Case
            if (OrderLs.index(k) in np.where(GetLs[:,0] != 0)[0]):
                ReBuildTrain(TmpIndex,Count,Train_Tags,Train_Labels,Prefix,Prediction,Title)
                Count += 1
                GetLs[OrderLs.index(k),0] -= 1
            else:
                RunSym1 = True
        else:
            #Neg Case
            if (OrderLs.index(k) in np.where(GetLs[:,1] != 0)[0]):
                ReBuildTrain(TmpIndex,Count,Train_Tags,Train_Labels,Prefix,Prediction,Title)
                Count += 1
                GetLs[OrderLs.index(k),1] -= 1
            else:
                RunSym1 = True
        TmpIndex += 1
        #if (OrderLs.index(k) not in np.where(GetLs[:,0] != 0)[0]) and (OrderLs.index(k) not in np.where(GetLs[:,0] != 0)[0]) 

#重新组合训练集
def ReBuildTrain(Index,Count,Tag,Label,Prefix,Prediction,Title):
    n_Prefix.append(Prefix[Index])
    n_Title.append(Title[Index])
    n_Tag.append(Tag[Index])
    n_Label.append(Label[Index])
    n_Prediction[str(Count)] = Prediction[str(Index)]

#Json Data Save To Json File
def JsonDataSave(path):
    with open(path+'OgeekMini.json','w',encoding='utf-8') as outfile:
        json.dump(n_Prediction,outfile,ensure_ascii=False)
        outfile.write('\n')

#Other Data Save To .mat
def Data2Mat(path):
    sio.savemat(path+'OGeekDataMini.mat',{'prefix':n_Prediction,'title':n_Title,'tag':n_Tag,'label':n_Label})

def main():
    #读取Train的信息
    Prefix,Prediction,Title,Tag,Label = input_data('OGeekData.mat','Oggek.json')
    #读取Mini Batch数量
    TagSplit(Tag,Label,Prefix,Prediction,Title)
    #存储变量
    JsonDataSave('')
    Data2Mat('')

main()