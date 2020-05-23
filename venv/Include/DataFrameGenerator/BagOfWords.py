import json
import numpy as np
import re
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math as mt




class Bow:
    def calculateWordsFromJson(self,path):
       list=[]
       data= json.load(open(path))
       for key in data:
           list.append(data[key])
       return self.sum(list)

    def createVector(self,spamDic,messages):
       # big_vector=np.zeros(len(spamDic))
        big_vector={}
        print('length of spamDic====',len(spamDic))
        for key in spamDic:
            big_vector.update({key:0})
        for message in messages:
            self.messageLenth=len(message)
            print('message Length====',self.messageLenth)
            for words in message:
                if words in big_vector:
                    big_vector[words]+=1

        return  sum(big_vector.values())/len(spamDic)+self.messageLenth

    def generateDic(self,path):
        return json.load(open(path))



    def calculateprobiltyofWord(words, documentDic,messageLength):
        return documentDic[words]/messageLength



    def create_doc_dic(self,message,vocSpam):
        documnet = vocSpam
        for w in message:
            if w in vocSpam:
                documnet[w]+=1
        return documnet


    def train(self,data):
        self.trainData=data
        text=data['messages']
        txt = []
        for tx in text:
            innertxt = []
            mee = re.sub('[^a-zA-Z0-9 \n\.]', '', tx)
            me = re.sub(r'[?|$|.|!]', r'', mee)
            strs = ''.join(c if c not in map(str, range(0, 10)) else '' for c in me)
            data = word_tokenize(strs.lower())
            sw = stopwords.words('english')
            Sw = [word for word in data if word not in sw]
            stemmer = PorterStemmer()
            words = [stemmer.stem(wo) for wo in Sw]
            innertxt.append(words)
            txt.append(innertxt)
        return self.predictionNew(txt)


    def createCompleteDoc_sampLength(self,vocSpam,message):
            documnet = vocSpam
            for w in message:
                if w in vocSpam:
                    documnet[w] += 1
                else:
                    documnet.update({w:0})

            return len(documnet)

    def predictionNew(self,data):
        vocSpam = json.load(open('F:/pythonWorkspace/SpamDetector/venv/Include/WordCountJson/spamCount.json'))
        vocHam = json.load(open('F:/pythonWorkspace/SpamDetector/venv/Include/WordCountJson/hamCount.json'))
        predictData = []
        for messages in data:

            for message in messages:
                documentDicSpam = self.create_doc_dic(message, vocSpam)
                documentDicHam = self.create_doc_dic(message, vocHam)
                self.totaldocumentsizeSpam=self.createCompleteDoc_sampLength(vocSpam,message)
                self.totaldocumentsizeHam=self.createCompleteDoc_sampLength(vocHam,message)
                prob_of_doc_in_spam_class = 0
                prob_of_doc_in_ham_class = 0
                for w in documentDicSpam.values():
                    prob_of_doc_in_spam_class +=mt.log((w+1),(len(vocSpam) + 1+self.totaldocumentsizeSpam))#log( (w + 1) / (len(vocSpam) + 1+self.totaldocumentsizeSpam))
                for w1 in documentDicHam.values():
                    prob_of_doc_in_ham_class +=mt.log((w1 + 1),(len(vocHam) + self.totaldocumentsizeHam + 1))#log((w1 + 1) / (len(vocHam) + self.totaldocumentsizeHam + 1))
                spamPredict=0
                spamPredict=  prob_of_doc_in_spam_class * len(vocSpam)/self.totaldocumentsizeSpam
                hamPredict=0
                hamPredict=   prob_of_doc_in_ham_class * len(vocHam)/self.totaldocumentsizeHam
                if spamPredict > hamPredict:
                    print("SpamPredict",spamPredict)
                    predictData.append('spam')
                elif spamPredict < hamPredict:
                    print("HamPredict",hamPredict)
                    predictData.append('ham')

        print("prdicted Data List Lenght", len(predictData))

        # df=pd.DataFrame(predictedData,columns=['Prediction'])
        self.trainData['Prediction'] = predictData
        return self.trainData









