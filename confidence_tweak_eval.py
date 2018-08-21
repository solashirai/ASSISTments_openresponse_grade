from __future__ import division, print_function, absolute_import
import os
import sys
import time
#test
import numpy as np
import data_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Lasso, RidgeClassifierCV, Ridge
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# disable tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_sample = 10

# print flags info
orig_stdout = sys.stdout
timestamp = str(int(time.time()))

training_path = 'filter_engageny_openresponse.csv'  # name of dataset
essay_list, label, problem_id, count_one, question_list = data_utils.load_open_response_data(training_path)
pidlist = list(set(problem_id))

root_dir = 'runs/newfilter_allproblems_threefeatures'

# go through each problem of interest

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].reshape(-1, 1)

if __name__ == '__main__':

    # generate and save a spreadsheet containing information about the count/accuracy/kappa for each problem
    # filtering out based on each prediction's decision function value.
    df = pd.DataFrame(columns=['PID',
                               '0.1 count', '0.1 acc', '0.1 kap',
                               '0.2 count', '0.2 acc', '0.2 kap',
                               '0.3 count', '0.3 acc', '0.3 kap',
                               '0.4 count', '0.4 acc', '0.4 kap',
                               '0.5 count', '0.5 acc', '0.5 kap',
                               '0.6 count', '0.6 acc', '0.6 kap',
                               '0.7 count', '0.7 acc', '0.7 kap',
                               '0.8 count', '0.8 acc', '0.8 kap',
                               '0.9 count', '0.9 acc', '0.9 kap'])

    for pid in pidlist:
        statlist = []
        predicted_lists = []
        true_lists = []
        training_path = root_dir+'/'+'test_details.csv' #name of dataset
        training_df = pd.read_csv(training_path, encoding='latin1')
        # get only rows for a specific problem id
        if pid != '':
            training_df = training_df[training_df['PID'].isin([pid])]
        if len(training_df) == 0:
            continue
        essay_list = []
        essays = training_df['Text']
        scores = training_df['PredictionCorrect'].tolist()
        decision_func1 = training_df['DF1'].tolist()
        decision_func2 = training_df['DF2'].tolist()
        decision_func3 = training_df['DF3'].tolist()
        decision_func4 = training_df['DF4'].tolist()
        decision_func5 = training_df['DF5'].tolist()
        decision_func = []
        for d in range(len(decision_func1)):
            dfuncs = [decision_func1[d], decision_func2[d], decision_func3[d], decision_func4[d], decision_func5[d]]
            decision_func.append(max(dfuncs))
        true_labels = training_df['Label'].tolist()
        predicted_labels = training_df['Predicted'].tolist()

        confidencelist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for conf in confidencelist:
            predicted_lists.append([])
            true_lists.append([])
            temp = []
            for f in range(len(decision_func)):
                if decision_func[f] > conf:
                    temp.append(scores[f])
                    predicted_lists[-1].append(predicted_labels[f])
                    true_lists[-1].append(true_labels[f])
            statlist.append(len(temp))
            statlist.append(np.mean(temp))
        df = df.append({'PID': pid,
                        '0.1 count': statlist[0],
                        '0.1 acc': statlist[1],
                        '0.1 kap': cohen_kappa_score(predicted_lists[0], true_lists[0]),
                        '0.2 count': statlist[2],
                        '0.2 acc': statlist[3],
                        '0.2 kap': cohen_kappa_score(predicted_lists[1], true_lists[1]),
                        '0.3 count': statlist[4],
                        '0.3 acc': statlist[5],
                        '0.3 kap': cohen_kappa_score(predicted_lists[2], true_lists[2]),
                        '0.4 count': statlist[6],
                        '0.4 acc': statlist[7],
                        '0.4 kap': cohen_kappa_score(predicted_lists[3], true_lists[3]),
                        '0.5 count': statlist[8],
                        '0.5 acc': statlist[9],
                        '0.5 kap': cohen_kappa_score(predicted_lists[4], true_lists[4]),
                        '0.6 count': statlist[10],
                        '0.6 acc': statlist[11],
                        '0.6 kap': cohen_kappa_score(predicted_lists[5], true_lists[5]),
                        '0.7 count': statlist[12],
                        '0.7 acc': statlist[13],
                        '0.7 kap': cohen_kappa_score(predicted_lists[6], true_lists[6]),
                        '0.8 count': statlist[14],
                        '0.8 acc': statlist[15],
                        '0.8 kap': cohen_kappa_score(predicted_lists[7], true_lists[7]),
                        '0.9 count': statlist[16],
                        '0.9 acc': statlist[17],
                        '0.9 kap': cohen_kappa_score(predicted_lists[8], true_lists[8])}, ignore_index=True)
    df.to_csv(root_dir+'/'+'confidencescore_evaluations.csv')

