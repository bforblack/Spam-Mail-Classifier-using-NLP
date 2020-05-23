from DataFrameGenerator import DataFrameGenerator
from MessagesCleaning import RemoveSpecilaCharater
import json
from BagOfWords import Bow as bow
import numpy as np
from VocabularyGenerator import VocabularyGenerator
from ConfusionMetrix import ConfusionMetrix
import pandas as pd
class Main:
    (trainData,testData,spam_data,ham_Data) = DataFrameGenerator('C:/Users/Sonal/Downloads/mailData/spam/*','C:/Users/Sonal/Downloads/mailData/easy_ham/*').generateDataFrame()
    print('trainData\n',len(trainData),'\nspam Data\n',len(spam_data),'\nHamData\n',len(ham_Data))
    spamvoc,hamvoc =VocabularyGenerator().prepare(trainData)
    hm=ham_Data[:len(spam_data)]
    testHamData=ham_Data[:3]
    testSpamData=spam_data[:3]
    print('hm',hm.head)
    newtestHam_spam=pd.concat([testHamData,testSpamData],axis=0)
    balancedDataset=pd.concat([spam_data,hm],axis=0)
    print('Blanaced Data',len(balancedDataset))
    print('Test Ham Spam',len(newtestHam_spam))
    prediction=bow().train(testData)
    #df.to_csv('F:/pythonWorkspace/SpamDetector/venv/Include/WordCountJson/prediction.csv', encoding='utf-8', index=False)
    prediction.to_csv('G:prediction2.csv', encoding='utf-8',index=False)

    print('predict HEad()\n',prediction.head(),'\n df sample\n',prediction.sample())

    ConfusionMetrix().createMetric(prediction)





