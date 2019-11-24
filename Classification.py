import numpy as np
from sklearn import datasets
import pandas as pd
from word2number import w2n
import nltk


def data_exploration():
    dataset = pd.read_csv("ner_dataset.csv")
    print("Number of tags", dataset['POS'].copy().drop_duplicates().count())
    print("Frequency of Values: ")
    print(dataset['POS'].copy().value_counts())


    patterns = [
        (r'.*ing', 'VBG'),
        (r'.*ed', 'VBD'),
        (r'.*es', 'VBZ'),
        (r'.*s', 'NNS'),
        (r'.*uld', 'MD'),
        (r'.*y', 'RB'),
        (r'.*', 'NN')
    ]

    rule_tagger = nltk.RegexpTagger(patterns)
    print(rule_tagger.tag(dataset['Word'][:50]))
    print("Accuracy, ", rule_tagger.evaluate((dataset['Word'], dataset['POS'])))

data_exploration()
