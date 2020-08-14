import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def get_total_number_of_facist_cards(players):
    return (players['pulled_3_facist'] * 3) + (players['pulled_2_facist_1_liberal'] * 2) + (players['pulled_2_liberal_1_facist'] * 1)

def get_total_number_of_liberal_cards(players):
    return (players['pulled_3_liberal'] * 3) + (players['pulled_2_liberal_1_facist'] * 2) + (players['pulled_2_facist_1_liberal'] * 1)

def fit_and_print_model(model, X, y):
    model.fit(X, y)
    print("{} ROC AUC score: {}".format(type(model), roc_auc_score(model.predict(X), y)))
    print("{} F1 score: {}".format(type(model), f1_score(model.predict(X), y)))

def grid_and_print_model(model, params, X, y):
    grid = GridSearchCV(model, params, cv=5, iid='deprecated')
    grid.fit(X, y)
    print("{} best score is {}".format(type(model), grid.best_score_))
    return grid

