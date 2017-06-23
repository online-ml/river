from skmultiflow.classification.naive_bayes import NaiveBayes
from sklearn.naive_bayes import MultinomialNB

def CreateStreamFromArgumentDict(argumentList):
    if argumentList[0] == 'NaiveBayes':
        if len(argumentList) > 1:
            return NaiveBayes(argumentList[1:])
        else:
            return NaiveBayes()
    return None