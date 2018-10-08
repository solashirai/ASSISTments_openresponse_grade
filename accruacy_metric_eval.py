from __future__ import division, print_function, absolute_import
import os
import sys
import time
#test
import numpy as np
import data_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
from math import sqrt, isnan
from scipy.stats import entropy

# disable tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_sample = 10

# print flags info
orig_stdout = sys.stdout
timestamp = str(int(time.time()))

#
training_path = 'Final_Filtered_Data_John_update.csv'  # name of dataset
pidlist = data_utils.load_pidlist(training_path)

data_dir = 'runs/jesus2_timestamp1539006507'
# go through each problem of interest


def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].reshape(-1, 1)

if __name__ == '__main__':

    df = pd.DataFrame(columns=['PID',
                               'Accuracy',
                               'Kappa',
                               'RMSE',
                               'Error Variance',
                               'AUC metric 1',
                               'AUC metric 2',
                               'AUC metric 3',
                               'AUC metric 4',
                               'AUC metric 5',
                               'AUC avg',
                               'Weighted F1 score'
                               ])

    for pid in pidlist:
        statlist = []
        predicted_lists = []
        true_lists = []
        training_path = data_dir+'/'+'test_details.csv' #name of dataset
        training_df = pd.read_csv(training_path, encoding='latin1')
        # get only rows for a specific problem id
        if pid != '':
            training_df = training_df[training_df['PID'].isin([pid])]
        if len(training_df) == 0:
            continue
        essay_list = []
        essays = training_df['Text']
        scores = training_df['PredictionCorrect'].tolist()
        true_labels = training_df['Label'].tolist()
        predicted_labels = training_df['Predicted'].tolist()
        maj_lab = most_common(true_labels)
        #predicted_labels = np.repeat(maj_lab, len(predicted_labels))
        dec_funs = [training_df['DF1'].tolist(),
               training_df['DF2'].tolist(),
               training_df['DF3'].tolist(),
               training_df['DF4'].tolist(),
               training_df['DF5'].tolist()]
        roc_auc_scores = ['', '', '', '', '']
        actual_auc_scores = []
        # spaghetti
        scores = []
        for s in range(5):
            if s in true_labels:
                scores.append(s)
        score_index = 0
        ### Calculating AUC
        for dec_fun in dec_funs:
            if len(scores) != 1:
                if not isnan(np.sum(dec_fun)):
                    if len(scores) == 2:
                        # when only two scores are present, flip score binarization, since the positive decision function
                        # would result in predicting the higher score
                        binarized_labels = np.where(np.array(true_labels) == scores[score_index], 0, 1)
                    else:
                        binarized_labels = np.where(np.array(true_labels) == scores[score_index], 1, 0)
                    aucscore = roc_auc_score(binarized_labels, dec_fun)
                    roc_auc_scores[scores[score_index]] = aucscore
                    actual_auc_scores.append(aucscore)
                    score_index += 1
                else:
                    roc_auc_scores.append('')
        roc_auc_avg = np.mean(actual_auc_scores)

        kap = cohen_kappa_score(true_labels, predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)

        # convert to int to perform rmse calculation
        true_labels_ints = [int(lab) for lab in true_labels]
        predicted_labels_ints = [int(lab) for lab in predicted_labels]
        rmse = sqrt(mean_squared_error(true_labels_ints, predicted_labels_ints))
        errors = np.array(true_labels_ints) - np.array(predicted_labels_ints)
        error_var = np.var(errors)

        df = df.append({'PID': pid,
                        'Accuracy': acc,
                        'Kappa': kap,
                        'RMSE': rmse,
                        'Error Variance': error_var,
                        'AUC metric 1': roc_auc_scores[0],
                        'AUC metric 2': roc_auc_scores[1],
                        'AUC metric 3': roc_auc_scores[2],
                        'AUC metric 4': roc_auc_scores[3],
                        'AUC metric 5': roc_auc_scores[4],
                        'AUC avg': roc_auc_avg,
                        'Weighted F1 score': f1_score(true_labels, predicted_labels, average='weighted')
                       }, ignore_index=True)
    df.to_csv(data_dir+'/'+'majorityclass_error_metric_evaluations.csv')

