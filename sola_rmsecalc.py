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
from collections import Counter
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

do_maj = False

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].reshape(-1, 1)

def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

if __name__ == '__main__':

    df = pd.DataFrame(columns=['PID',
                               'RMSE avg',
                               'RMSE 1',
                               'RMSE 2',
                               'RMSE 3',
                               'RMSE 4',
                               'RMSE 5',
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
        maj_score = most_common(true_labels)
        predicted_labels = training_df['Predicted'].tolist()
        dec_funs_raw = [
                training_df['DF1'],
               training_df['DF2'],
               training_df['DF3'],
               training_df['DF4'],
               training_df['DF5']]
        dec_funs = []
        for decf in dec_funs_raw:
            dec_funs.append(np.nan_to_num(decf))
        RMSEs = [0, 0, 0, 0, 0]
        # spaghetti
        scores = []
        for s in range(5):
            if s in true_labels:
                scores.append(s)
        score_index = 0
        ### Calculating RMSEs
        df_ind = 0
        for dec_fun in dec_funs:
            if do_maj:
                if df_ind == maj_score:
                    dec_fun = np.ones(len(true_labels))
                else:
                    dec_fun = np.zeros(len(true_labels))
            if df_ind == scores[score_index]:
                binarized_labels = np.where(np.array(true_labels) == scores[score_index], 1, 0)
                rmse = mean_squared_error(binarized_labels, dec_fun) #** 0.5
                RMSEs[scores[score_index]] = rmse
                score_index += 1
            df_ind += 1

        df = df.append({'PID': pid,
                        'RMSE avg': np.mean(RMSEs),
                        'RMSE 1': RMSEs[0],
                        'RMSE 2': RMSEs[1],
                        'RMSE 3': RMSEs[2],
                        'RMSE 4': RMSEs[3],
                        'RMSE 5': RMSEs[4]
                       }, ignore_index=True)
    df.to_csv(data_dir+'/'+'fferror_metric_evaluations.csv')

