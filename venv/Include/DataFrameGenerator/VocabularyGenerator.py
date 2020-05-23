from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import re
class VocabularyGenerator:

    def prepare(self,trainData):
        self.spamData= trainData.loc[trainData['type'] == 'spam']
        self.hamData= trainData.loc[trainData['type'] == 'ham']
        self. cleanedSpam=self.dataCleaning(self.spamData)
        self.cleanedHam=self.dataCleaning(self.hamData)
        self.generateVocablaryJson()
        print('Vocablory  Generated SucessFully')
        return self.cleanedSpam,self.cleanedHam



    def dataCleaning(self,Data):
        messages = Data['messages']
        self.mess = []
        for message in messages:
            innermess = []
            me=re.sub('[^a-zA-Z0-9 \n\.]', '', message)
            me = re.sub(r'[?|$|.|!]', r'', me)
            strs = ''.join(c if c not in map(str, range(0, 10)) else '' for c in me)
            data = word_tokenize(strs.lower())
            sw = stopwords.words('english')
            Sw = [word for word in data if word not in sw]
            stemmer = PorterStemmer()
            words = [stemmer.stem(wo) for wo in Sw]
            innermess.append(words)
            self.mess.append(innermess)

        return self.generatedmap()

    def generateVocablaryJson(self):
            with open('F:/pythonWorkspace/SpamDetector/venv/Include/WordCountJson/spamCount.json', 'w') as fp:
                json.dump(self.cleanedSpam, fp, sort_keys=True, indent=4)
            with open('F:/pythonWorkspace/SpamDetector/venv/Include/WordCountJson/hamCount.json', 'w') as fp:
                json.dump(self.cleanedHam, fp, sort_keys=True, indent=4)

    def generatedmap(self):
        bigvect={}
        for mes in self.mess:
            for me in mes:
                for m in me:
                    bigvect.update({m: 0})

        return bigvect









