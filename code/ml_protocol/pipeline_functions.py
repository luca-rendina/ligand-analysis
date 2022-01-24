from utility_functions import *
from ensemble_functions import *

import random
from random import seed 
import glob
import os

import pandas as pd
import numpy as np

from openbabel import openbabel

import sklearn.metrics
from sklearn.metrics import roc_curve, auc, make_scorer, confusion_matrix, roc_auc_score
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree

from IPython.display import Image

import pydotplus

import oddt
from oddt.fingerprints import InteractionFingerprint, PLEC, SPLIF, ECFP

import joblib
import json

# matplotlib
import matplotlib.pyplot as plt

pattern_agonist = ["[0-9]_full_agonist", "[0-9]_agonist"]
pattern_antagonist = ["antagonist"]
DATA_folder = "DATAS/"
output_folder = "ROC/"

def preprocessing(uniprot_id, pdb_id):

    file_agonist = [file for p in pattern_agonist for file in glob.glob(DATA_folder+"Ligands/"+pdb_id+"*"+p+"*best.sdf")]
    file_antagonist = [file for p in pattern_antagonist for file in glob.glob(DATA_folder+"Ligands/"+pdb_id+"*"+p+"*best.sdf")]
    
    # Save the image of each agonist
    agonist_poses_tmp = []
    for file in file_agonist:
        agonist_poses_tmp.append(list(oddt.toolkit.readfile("sdf", "{}".format(file))))

    # Save the image of each antagonist
    antagonist_poses_tmp = []
    for file in file_antagonist:
        antagonist_poses_tmp.append(list(oddt.toolkit.readfile('sdf', '{}'.format(file))))

    # Create a duplicate of each list
    antagonist_poses = [x[:] for x in antagonist_poses_tmp]
    agonist_poses = [x[:] for x in agonist_poses_tmp]

    # read the protein
    protein = next(oddt.toolkit.readfile('pdbqt', DATA_folder+pdb_id+'_'+uniprot_id+'_proc_cleaned.pdbqt'))
    protein.protein = True
    
    # list of fingerprint for all agonist_poses best protein_depth=4 ligand_depth=2 size=65536
    IFP_list_agonists = np.array([PLEC(x[0], protein, sparse=False, depth_protein=4, depth_ligand=2, size=65536) for x in agonist_poses])
    # list of fingerprint for all antagonist_poses
    IFP_list_antagonists = np.array([PLEC(x[0], protein, sparse=False, depth_protein=4, depth_ligand=2, size=65536) for x in antagonist_poses])
    
    #add 1 if class antagonist
    antagonist_with_active = []
    for subarray in IFP_list_antagonists:
        antagonist_with_active.append(np.append([subarray],[1]))
    #add 0 if class agonist
    agonist_with_active = []
    for subarray in IFP_list_agonists:
        agonist_with_active.append(np.append([subarray],[0]))

    # put all agonist and antagonist inside data
    data = []
    for e in agonist_with_active:
        data.append(e)
    for e in antagonist_with_active:
        data.append(e)
    
    FEATURES = np.array([data[i][:-1:] for i in range(len(data))])
    LABELS = np.array([data[i][-1] for i in range(len(data))])
    
    return FEATURES, LABELS

def save_log(classifiers_normalized, uniprot_id, pdb_id):
    name_file = output_folder+uniprot_id+"_"+pdb_id+".log"
    log_models = open(name_file, "w")
    for c, cm in classifiers_normalized:
        log_models.write("{'"+c.__class__.__name__+"':")
        log_models.write(json.dumps(measure("all", cm), sort_keys=True, indent=4)+"\n")
        log_models.write("}")
    log_models.close()
    return name_file
    
def save_score(cm_ensemble, uniprot_id, pdb_id):
    scores = measure("all", cm_ensemble)
    name_file = output_folder+uniprot_id+"_"+pdb_id+"_ensembl.json"
    with open(name_file, "w") as fp:
        json.dump(scores, fp, sort_keys=True, indent=4)
    return name_file

def save_roc(classifiers_normalized, TEST, TEST_LABELS, uniprot_id, pdb_id):
    pred_prob = []
    for m, cm in classifiers_normalized:
        pred_prob.append([m.__class__.__name__, m.predict(TEST)])

    # roc curve for models
    info = []
    for n, pred in pred_prob:
        info.append([n, roc_curve(TEST_LABELS, pred, pos_label=1), roc_auc_score(TEST_LABELS, pred)])

    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(TEST_LABELS))]
    p_fpr, p_tpr, _ = roc_curve(TEST_LABELS, random_probs, pos_label=1)
    
    # auc scores
    auc_score = []
    for _, _,score  in info:
        auc_score.append(score)
    
    plt.clf()
    
    plt.style.use('seaborn')
    # plot roc curves
    for n, i,score  in info:
        plt.plot(i[0], i[1], linestyle='--', label=n+' (auc %0.2f)' % score)

    # title
    plt.title(uniprot_id+"_"+pdb_id)
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(output_folder+uniprot_id+'_'+pdb_id+'_ROC',dpi=300)
    
    return plt

def save_model(model, uniprot_id, pdb_id):
    joblib_file = output_folder+uniprot_id+"_"+pdb_id+"_joblib.pkl"  
    joblib.dump(model, joblib_file)
    return True

def load_model(folder, uniprot_id, pdb_id):
    joblib_file = output_folder+uniprot_id+"_"+pdb_id+"_joblib.pkl"
    joblib_LR_model = joblib.load(joblib_file)
    return joblib_LR_model
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    