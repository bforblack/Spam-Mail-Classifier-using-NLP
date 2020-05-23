from glob import glob
import pandas as pd
import numpy as np


class DataFrameGenerator:

    def __init__(self,spam_path,ham_path):
        self.spam = []
        self.ham = []
        spam_files=glob(spam_path)
        ham_files=glob(ham_path)
        for file in spam_files:
            innerSpam=[]
            innerSpam.append(open(file,'r',encoding='latin-1').read())
            innerSpam.append('spam')
            self.spam.append(innerSpam)


        for files in ham_files:
            inner_ham=[]
            inner_ham.append(open(files,'r',encoding='latin-1').read())
            inner_ham.append('ham')
            self.ham.append(inner_ham)



    def generateDataFrame(self):
        df=pd.DataFrame(self.spam,columns=['messages','type'])
        df2=pd.DataFrame(self.ham,columns=['messages','type'])
        finalDataSet=pd.concat([df,df2],axis=0)
        trainData=finalDataSet.sample(frac=0.7)
        testData=finalDataSet.sample(frac=0.3)
        return(trainData,testData,trainData.loc[trainData['type']=='spam'],trainData.loc[trainData['type']=='ham'])










