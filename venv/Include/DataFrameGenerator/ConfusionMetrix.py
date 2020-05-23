class ConfusionMetrix:
    def createMetric(self,predictedData):
        true_possitive,true_negative,false_possitive,false_negative=0,0,0,0
        oreginalData,verifyData=predictedData['type'],predictedData['Prediction']
        for i in range(len(predictedData)):
            true_possitive+=oreginalData.iloc[i]=='ham' and verifyData.iloc[i]=='ham'
            true_negative+=oreginalData.iloc[i]=='spam' and verifyData.iloc[i]=='spam'
            false_possitive+=oreginalData.iloc[i]=='spam' and verifyData.iloc[i]=='ham'
            false_negative+=oreginalData.iloc[i]=='ham' and verifyData.iloc[i]=='spam'

        precission_call=true_possitive/(true_possitive+false_possitive)
        recall=true_possitive/(true_possitive+false_negative)
        f1_score=2*precission_call*recall/(precission_call+recall)
        accuracy=(true_negative+true_possitive)/(true_possitive+true_negative+false_possitive+false_negative)

        print('precission_call',precission_call)
        print('recall', recall)
        print('f1_score', f1_score)
        print('accuracy', accuracy)
