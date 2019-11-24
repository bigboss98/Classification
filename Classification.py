import numpy as np
from sklearn import datasets
import pandas as pd
from word2number import w2n
import nltk


def data_exploration():
    dataset = pd.read_csv("ner_dataset.csv")
    for col in dataset.columns:
        print(col)
        print(dataset[col].copy().drop_duplicates().values)
        print(dataset[col].copy().value_counts())

    for value in dataset['Word']:
        try:
            if w2n.word_to_num(value):
                print("CD")
        except ValueError:
            pass

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
