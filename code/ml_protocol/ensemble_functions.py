import random
from random import seed

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_validate, cross_val_score
from sklearn.metrics import fbeta_score, make_scorer, confusion_matrix
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.svm import SVC

#Visualization
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

from utility_functions import *

class ensemble_model():

    random_state = 0

    ClassifierList = [ [RandomForestClassifier, {"n_estimators": 5,
                                                 "criterion": "entropy",
                                                 "min_samples_split": 10,
                                                 "max_depth": 8,
                                                 "random_state": random_state}] , 
                      [GradientBoostingClassifier, {'criterion': 'friedman_mse', 
                                                    'learning_rate': 0.1, 
                                                    'loss': 'exponential', 
                                                    'max_depth': 3, 
                                                    'min_samples_split': 10, 
                                                    'n_estimators': 10,
                                                    "random_state": random_state}], 
                      [SGDClassifier, {"alpha": 1e-4,
                                       "max_iter": 100, 
                                       "random_state": random_state}],
                      [SVC, {"random_state": random_state, "kernel": "linear", "C": 0.025}],
                      [GaussianProcessClassifier, {"kernel": 1*DotProduct(),
                          "random_state": random_state}],
                      #[XGBClassifier, {"random_state": random_state}],
                      #[MLPClassifier, {"solver":'adam', 
                      #                 "alpha": 1e-02, 
                      #                 "max_iter": 100, 
                      #                 "hidden_layer_sizes": (20), 
                      #                 "random_state": random_state}]
                     ]  

    
    """ Splitting FEATURES and LABELS in TRAIN and TEST """
    
    def split_data_train_test(self, FEATURES, LABELS, ratio = 0.3):

        #create list with the division between the occurency of the event

        no_event_train, no_event_test, no_event_labels_train, no_event_labels_test = train_test_split(FEATURES[LABELS == 0], 
                                                                                    LABELS[LABELS == 0], 
                                                                                    test_size=ratio, 
                                                                                    random_state=self.random_state)
        event_train, event_test, event_labels_train, event_labels_test = train_test_split(FEATURES[LABELS == 1],
                                                                              LABELS[LABELS == 1], 
                                                                              test_size=ratio, 
                                                                              random_state=self.random_state)
        #pre processing of TRAIN and TEST
        TRAIN_tmp = np.concatenate((no_event_train,event_train))
        TEST_tmp = np.concatenate((no_event_test,event_test))
        TRAIN_LABELS_tmp = np.concatenate((no_event_labels_train,event_labels_train))
        TEST_LABELS_tmp = np.concatenate((no_event_labels_test,event_labels_test))

        #TRAIN AND TEST LISTS
        TRAIN, TRAIN_LABELS = shuffle(TRAIN_tmp, TRAIN_LABELS_tmp)
        TEST, TEST_LABELS = shuffle(TEST_tmp, TEST_LABELS_tmp)
        
        return [TRAIN, TRAIN_LABELS, TEST, TEST_LABELS]
        
    """ Leave one out to check the accuracy of the different models in ClassifierList """
        
    def model_fit_leaveOneOut(self, TRAIN, TRAIN_LABELS, norm = True):
        #LIST OF CLASSIFIERS
        classifiers = []
        for c, v in self.ClassifierList:
            classifiers.append(c(**v))

        #LIST OF CLASSIFIERS with confusion matrix using leaveOneOut function
        classifiers_with_cm = []
        for c in classifiers:
            #cm = {"TP":0, "FP":0, 
            #      "FN":0, "TN":0}
            cm = np.array([[0,0], 
                          [0,0]])
            #print(c.__class__.__name__)

            loo = LeaveOneOut()
            for idx, i in loo.split(TRAIN):

                c = c.fit(TRAIN[idx], TRAIN_LABELS[idx])

                p = c.predict(TRAIN[i].reshape(1, -1))
                j = int(TRAIN_LABELS[i])
                cm[j][p] += 1

            classifiers_with_cm.append([c, cm])

        if norm:
            #normalize data of confusion matrix
            classifiers_normalized = []
            for c, cm in classifiers_with_cm:
                classifiers_normalized.append([c, normalize_cm(cm)])

        return classifiers_normalized

    """ Filtering the models returned from previous method given treshold and metric """

    def filter_ensemble(self, classifiers_normalized, n_selected = 0, treshold = 0.6, metric = "accuracy"):

        #FILTERING classifiers by Treshold of TargetMatrix
        idx_metrics = []
        for i in range(0, len(classifiers_normalized)):
            value = measure(metric, classifiers_normalized[i][1])
            if (value >= treshold):
                idx_metrics.append([i, value])

        #sort so we can take by MaxNumber
        idx_metrics.sort(reverse = True, key=mysort)

        if n_selected != 0:
            idx = np.array([k for k, v in idx_metrics[0:n_selected]])
        else:
            idx = np.array([k for k, v in idx_metrics])

        idx.sort()
        
        if(len(idx) == 0): #check if there are no classifiers that satisfy the treshold
            return np.array([])
        
        return np.array(classifiers_normalized)[idx]

    """ fitting the TRAINING data to the models """

    def fit_ensemble(self, target_classifiers, TRAIN, TRAIN_LABELS):

        #training the selected classifiers
        ensemble_model = []
        for c, cm in target_classifiers:
            ensemble_model.append([c.fit(TRAIN, TRAIN_LABELS), cm])
        return ensemble_model

    """ predicting by ensemble voting (sum the values and get the higher one)"""    

    def predict_ensemble(self, ensemble_model, TEST, TEST_LABELS):

        #TESTING with creation of confusion matrix
        cm_ensemble = [[0,0],[0,0]]
        if(len(ensemble_model) == 0): #if there are no classifier on the ensemble model
            return cm_ensemble
        
        for i in range(0, len(TEST)):
            target_cm = []
            for m, cm in ensemble_model:
                idx = m.predict(TEST[i].reshape(1, -1))
                target_cm.append(cm[idx])
            winner = np.argmax(sum(target_cm))
            cm_ensemble[TEST_LABELS[i]][winner] += 1
        return cm_ensemble