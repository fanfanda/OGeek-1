#OGeek的NTR数据解析 Lzz&Xlxw
#Ver In 2018.9.27
#1 Modify Code Issue
#2 Transfer To Static Class Method
#-----------------------------------------------------------------------------

#Lib Import
import json
import scipy.io as sio
import scipy.io as scio

#Data Struct
class OGeekData:
    def __init__(self):
        self.prefix             = []     #Query Prefix
        self.query_prediction   = {}     #P(Prediction)
        self.title              = []     #Passage Title
        self.tag                = []     #Passage Tag
        self.label              = []     #Whether Click

#Parser Method Class

class OGeekParser:
    #Data Parser
    @staticmethod
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
            TmpRSpace = TmpJson.replace(' ','')
            #Error Raise For Empty Data
            if TmpRSpace == "":
                continue
            Data.prefix.append(TmpData[0])
            Data.title.append(TmpData[2])
            Data.tag.append(TmpData[3])
            Data.label.append(TmpData[4])
            JsonData = json.loads(TmpRSpace)
            Data.query_prediction[JsonIndex] = JsonData
            JsonIndex = JsonIndex + 1
        #Close Txt
        File.close()
        return Data

    #Json Data Save To Json File
    @staticmethod
    def JsonDataSave(data,path):
        with open(path+'Ogeek.json','w',encoding='utf-8') as outfile:
            json.dump(data,outfile,ensure_ascii=False)
            outfile.write('\n')

    #Other Data Save To .mat
    @staticmethod
    def Data2Mat(data,path):
        sio.savemat(path+'OGeekData.mat',{'prefix':data.prefix,'title':data.title,'tag':data.tag,'label':data.label})

    #Use For Data OutPut
    @staticmethod
    def output_data(Source_path,Json_path,Mat_path):
        OGeekDataSet = OGeekData()
        OGeekDataSet = OGeekParser.Data2OGeek(Source_path)
        OGeekParser.JsonDataSave(OGeekDataSet.query_prediction,Json_path)
        OGeekParser.Data2Mat(OGeekDataSet,Mat_path)

    #Use For Data Input
    @staticmethod
    def input_data(MatFile_path,JsonFile_path):
        MatFile = scio.loadmat(MatFile_path)
        prefix = MatFile['prefix']
        title = MatFile['title']
        tag = MatFile['tag']
        label = MatFile['label']

        JsonFile = open(JsonFile_path,encoding='utf-8')
        prediction = json.load(JsonFile)
        return prefix, prediction, title, tag, label
             
    
#Main Func
if __name__ == '__main__':
    #Save File Method
    OGeekParser.output_data('oppo_round1_vali_20180926.txt','','')
    #Load File Method
    prefix, prediction, title, tag, label = OGeekParser.input_data('OGeekData.mat','Ogeek.json')
    


        
