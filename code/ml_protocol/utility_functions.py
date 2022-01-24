import numpy as np
import pandas as pd

#given two arrays, one with the values to change (to 0s and 1s) and the one with [PositiveValues]
def transform(array, PositiveValues):
    result = []
    for l in array:
        flag = False
        for v in PositiveValues:
            if l == v:
                flag = True
        if flag:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

#Return the classes which has at least half samples that will become our event
def select_positive_values(array):
    
    classes = np.array(pd.Series(array).drop_duplicates())
    
    if len(classes) <= 3:
        return classes[0]
    
    idx_class = [[ sum(classes[i] == array), i] for i in range(len(classes))]
    idx_class.sort(reverse = True)

    peak = len(array)/2
    grow = 0
    for N in range(len(idx_class)):
        grow = grow + idx_class[N][0]
        if grow/peak >= 1:
            return classes[np.array([i for c, i in idx_class[:N]])]
    

#FEATURES WITHOUT NILL ('?')
def idx_without_char(list_array, char='?'):
    idx = []
    
    for e in range(len(list_array)):
        if sum(list_array[e] == char) == 0:
            idx.append(e)
    idx = np.array(idx)
    return idx

def normalize_cm(cm, fix=2):
    n_classes = len(cm)
    cm = cm.astype(float)
    for i in range(0, n_classes):
        tot = sum(cm[i])
        for j in range(0, n_classes):
            cm[i][j] = round(cm[i][j] / tot, fix)
    return cm

def mysort(e):
    return(e[1])

def div0(n, d):
    return n / d if d else 0

def accuracy(cm): #TP + TN / (TP + TN + FP + FN)
    return(div0((cm[0][0] + cm[1][1]), (sum(cm[0]) + sum(cm[1]))))

def sensitivity(cm): #TP[00]o[11] / TP[00]o[11] + FN[10]o[01] 
    sensitivity = np.array([])
    for i in range(len(cm)):
        sensitivity = np.append(sensitivity, [div0(cm[i][i], sum([e[i] for e in cm]))])
    return(sensitivity)
    
def fp_rate(cm): #FP / FP + TN
    return(div0(cm[0][1], sum([e[1] for e in cm])))

def fn_rate(cm): #FN / FN + TP
    return(div0(cm[1][0], sum([e[0] for e in cm])))
    
def tn_rate(cm): #TN / TN + FP
    return(div0(cm[1][1], sum([e[1] for e in cm])))
    
def precision(cm): #TP[00] / TP[00] + FP[01]
    return(div0(cm[0][0], sum(cm[0])))
    
def f1_score(cm): #2*(precision*sensitivity / precision+sensitivity)
    scores = [precision(cm), sensitivity(cm)[0]]
    return(2 * div0(( scores[0] * scores[1]), (scores[0] + scores[1])))
    
def error_rate(cm): #FP + FN / P + N
    return(div0((cm[0][1] + cm[1][0]), (sum(cm[0]) + sum(cm[1]))))

def balanced_accuracy(cm):
    scores = [tn_rate(cm), sensitivity(cm)[0]]
    return((scores[1] + scores[0]) / 2)
    
def matthews_correlation(cm):
    return(div0( (cm[0][0]*cm[1][1] - cm[0][1]*cm[1][0]), np.sqrt( sum(cm[0] ) * sum([ e[0] for e in cm ]) * sum([ e[1] for e in cm ]) * sum(cm[1])) ))
    

def measure(target, cm):
    
    #cm = {"TP":00, "FP":01, 
    #      "FN":10, "TN":11} class 0
    
    #cm = {"TN":00, "FN":01, 
    #      "FP":10, "TP":11} class 1

    scores = {"sensitivity": sensitivity(cm)[0],
             "fp_rate": fp_rate(cm),
             "fn_rate": fn_rate(cm),
             "tn_rate": tn_rate(cm),
             "precision": precision(cm),
             "accuracy": accuracy(cm),
             "f1_score": f1_score(cm),
             "error_rate": error_rate(cm),
             "balanced_accuracy": balanced_accuracy(cm),
             "matthews_correlation": matthews_correlation(cm)}        
    
    if target != "all":
        return(scores[target])
    
    else:
        return(scores)