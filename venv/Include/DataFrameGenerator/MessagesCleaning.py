from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
class RemoveSpecilaCharater:

    def clean(self,messages):
        cleanedList=[]
        for message in messages:
            for char in ".,:/n":
                message.replace(char,'')
                message.lower()
                cleanedList.append[message.split()]
        return cleanedList


    def tokenizeData(self,Data):
        message = Data['messages']
        mess=[]
        for me in message:
            innermess=[]
            #for char in '<>[]{}()#?!;':
                #me=me.replace(char,'')
            data=word_tokenize(me.lower())
            sw = stopwords.words('english')
            Sw = [word for word in data if word not in sw]
            stemmer = PorterStemmer()
            words = [stemmer.stem(wo) for wo in Sw]
            innermess.append(words)
            mess.append(innermess)
            dict= self.countWordCount(mess)
        return (pd.DataFrame(mess,columns=['messages']),dict)

    def countWordCount(self,mess):
        d=dict()
        for message in mess:
            for words in message:
                for word in words:
                    if word in d:
                        d[word]=d[word]+1
                    else:
                        d[word] = 1
        return d

    def tokenizeSingleData(self, Data):
        message = Data['messages']
        mess = []
        for me in message:
            innermess = []
            # for char in '<>[]{}()#?!;':
            # me=me.replace(char,'')
            data = word_tokenize(me.lower())
            sw = stopwords.words('english')
            Sw = [word for word in data if word not in sw]
            stemmer = PorterStemmer()
            words = [stemmer.stem(wo) for wo in Sw]
            innermess.append(words)
            mess.append(innermess)
        return mess
