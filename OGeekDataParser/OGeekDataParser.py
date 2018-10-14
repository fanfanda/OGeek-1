#OGeek的NTR数据解析 Lzz&Xlxw
#2018.9.27
#1 Modify Code Issue
#-----------------------------------------------------------------------------

#Lib Import
import json
import scipy.io as sio

#Data Struct
class OGeekData:
    def __init__(self):
        self.prefix             = []     #Query Prefix
        self.query_prediction   = {}     #P(Prediction)
        self.title              = []     #Passage Title
        self.tag                = []     #Passage Tag
        self.label              = []     #Whether Click

#Data Parser
def Data2OGeek(path):
    #Create OGeekData Structer
    Data = OGeekData()
    #Open TXT
    File = open(path,'rb')
    #Var For Json Index
    JsonIndex = 0
    for line in File:
        #Read Every Lines
        TmpBinary = line.decode('utf-8')
        TmpData = TmpBinary.split('\t')
        TmpJson = TmpData[1]
        Data.prefix.append(TmpData[0])
        Data.title.append(TmpData[2])
        Data.tag.append(TmpData[3])
        Data.label.append(TmpData[4])
        TmpRSpace = TmpJson.replace(' ','')
        #Error Raise For Empty Data
        if TmpRSpace == "":
            continue
        JsonData = json.loads(TmpRSpace)
        Data.query_prediction[JsonIndex] = JsonData
        JsonIndex = JsonIndex + 1
    #Close Txt
    File.close()
    return Data
    

#Json Data Save To Json File
def JsonDataSave(data,path):
    with open(path+'Oggek.json','w',encoding='utf-8') as outfile:
        json.dump(data,outfile,ensure_ascii=False)
        outfile.write('\n')

#Other Data Save To .mat
def Data2Mat(data,path):
    sio.savemat(path+'OGeekData.mat',{'prefix':data.prefix,'title':data.title,'tag':data.tag,'label':data.label})
        
    
#Main Func
def main():
    OGeekDataSet = Data2OGeek('../data/oppo/oppo_round1_vali_20180929.txt')
    JsonDataSave(OGeekDataSet.query_prediction,'/')
    Data2Mat(OGeekDataSet,'/')
    
main()

        
