import numpy as np
import sklearn
import pandas as pd
from word2number import w2n
import nltk


def data_exploration():
    dataset = pd.read_csv("ner_dataset.csv")
    print("Number of tags", dataset['POS'].copy().drop_duplicates().count())
    print("Frequency of Values: ")
    print(dataset['POS'].copy().value_counts())

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
    print("Accuracy, ", sklearn.metrics.accuracy_score(dataset['POS'], [x for (_, x) in solutions]))

data_exploration()
