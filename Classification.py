import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from word2number import w2n
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier


def data_exploration():
    dataset = pd.read_csv("ner_dataset.csv")
    print("\nNumber of tags:", dataset['POS'].copy().drop_duplicates().count())
    print("\nTag Frequency: ")
    print(dataset['POS'].copy().value_counts())
    print('\nWord Frequency:')
    print(dataset.copy().groupby(["POS", "Word"])['Word'].count().sort_values().groupby(level=0).tail(1))

    for item in range(0, len(dataset['Word']) - 1):
        try:
            if w2n.word_to_num(dataset['Word'][item]):
                value = w2n.word_to_num(dataset['Word'][item])
                dataset['Word'][item] = str(value)
        except ValueError:
            pass

    patterns = [
        (r'.*ing', 'VBG'),
        (r'.*ed', 'VBD'),
        (r'.*es', 'VBZ'),
        (r'.*s', 'NNS'),
        (r'.*uld', 'MD'),
        (r'.*y', 'RB'),
        (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),
        (r'.*est', 'JJS'),
        (r'.*', 'NN')
    ]

    rule_tagger = nltk.RegexpTagger(patterns)
    solutions = rule_tagger.tag(dataset['Word'])
    print("Accuracy: ", sklearn.metrics.accuracy_score(dataset['POS'], [x for (_, x) in solutions]))
    del dataset
    del solutions


def obtain_features(word):
    features = {}
    features['lastCharachter'] = str(word[-1:])
    features['last2characher'] = str(word[-2:])
    features['last3characher'] = str(word[-3:])
    return features


def obtain_training_set(dataset):
    return [(obtain_features(dataset['Word'][index]), dataset['POS'][index]) for index in range(0, len(dataset))]

def obtain_testset(dataset):
    return [obtain_features(word) for word in dataset['Word']]

def classification(value):
    trainingSet = pd.read_csv("ner_dataset.csv")
    dataSet = pd.read_csv("ner_test.csv")
    print("Done it")
    featureSet = obtain_training_set(trainingSet)
    testSet = obtain_testset(dataSet)

    #classifier = nltk.DecisionTreeClassifier.train(featureSet[:5000])
    #accuracy1 = nltk.classify.accuracy(classifier, featureSet[size:])
    #print("Accuracy of Decision Tree classifier: ", accuracy1)
    #secondClassifier = nltk.NaiveBayesClassifier.train(featureSet[:5000])
    #accuracy2 = nltk.classify.accuracy(secondClassifier, featureSet[size:])
    #print("Accuracy of Naive Bayes Classifier: ", accuracy2)
    #thirdClassifier = SklearnClassifier(KNeighborsClassifier()).train(featureSet[:5000])
    #accuracy3 = nltk.classify.accuracy(thirdClassifier, featureSet[size:])
    #print("Accuracy of K-neighbour classifier", accuracy3)
    # information = CountVectorizer(analyzer=obtain_features(trainingSet, value), lowercase=False)
    # information.fit_transform(trainingSet['Word']).toarray()
    firstClassifier = SklearnClassifier(SGDClassifier()).train(featureSet)
    accuracy4 = nltk.classify.accuracy(firstClassifier, featureSet)
    print("Accuracy of linear model", accuracy4)
    solution = []
    for index in range(0, len(dataSet['Word'])):
        predictTag = str(firstClassifier.classify(testSet[index]))
        solution.append((dataSet['Word'][index], predictTag))
    return 0, 0, 0, accuracy4, solution


data_exploration()

#for val in values:
 #   val1, val2, val3, val4 = classification(val)
  #  plot_value[0].append(val1)
   # plot_value[1].append(val2)
    #plot_value[2].append(val3)
    #plot_value[3].append(val4)

val1, val2, val3, val4, solution = classification(250)

with open('solution_group8.csv', 'w') as csvfile:
    headerwriter = csv.DictWriter(csvfile, fieldnames=['Index', 'POS'])
    headerwriter.writeheader()
    writer = csv.writer(csvfile, delimiter=",")

    index = 0
    listLines = []
    for (word, tag) in solution:
        line = [index, tag]
        index += 1
        listLines.append(line)
    writer.writerows(listLines)
csvfile.close()
