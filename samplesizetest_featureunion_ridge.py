from __future__ import division, print_function, absolute_import
import os
import re
import sys
import time
#test
import numpy as np
import data_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Lasso, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
from multiprocessing import Pool, Lock
from contextlib import closing
import itertools
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats import entropy


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# disable tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_sample = 10

# print flags info
orig_stdout = sys.stdout
timestamp = str(int(time.time()))

training_path = 'Final_Filtered_Data_John_update.csv'

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key != 'text':
            return data_dict[self.key].reshape(-1, 1)
        return data_dict[self.key]

def get_cosine_sims(essay_list, label, n):
    '''
    get cosine similarity of each sentence to sentences that recieved full scores.
    an average vector of sentences that recieved full scores is used.
    for a sentence that did receive a full score, its similarity to the rest of the full score sentences is calculated
    '''
    cosine_similarities_temp = []
    holdout_similarity_temp = 0
    for x in range(len(essay_list)):
        fullscore_essays = []
        fullscore_indices = []
        for y in range(len(essay_list)):
            if x != y and label[y] == 4:
                fullscore_indices.append(y)
                fullscore_essays.append(essay_list[y])
        # if no fullscore essays exist...
        if len(fullscore_essays) == 0:
            cosine_similarities_temp.append(0)
        else:
            vectorizer_temp = TfidfVectorizer(tokenizer=TreebankWordTokenizer().tokenize, analyzer='char',
                                              ngram_range=(2, n))
            try:
                vectorizer_temp.fit(fullscore_essays)

                vec_essays_temp = vectorizer_temp.transform(essay_list)
                top_vecs_temp = np.zeros(vec_essays_temp[0].shape)
                for fullscore_i in fullscore_indices:
                    top_vecs_temp += vec_essays_temp[fullscore_i]
                top_vecs_temp = top_vecs_temp / len(fullscore_indices)
                cosine_similarities_temp.append((vec_essays_temp[x] * top_vecs_temp.T).A[0][0])
            except ValueError:
                cosine_similarities_temp.append(0)
    return cosine_similarities_temp

# mostly just set up as a separate function to allow mutithreading
def run_model_on_problem(essay_list, label):


    # get word count for each student response
    wordcount = []
    for i in range(len(essay_list)):
        wordcount.append(len(essay_list[i]))
        essay_list[i] = ' '.join(word for word in essay_list[i])
    wordcount = np.array(wordcount)
    # wordcount = wordcount / np.max(wordcount)

    all_cosine_sims = [
        get_cosine_sims(essay_list, label, 3)
    ]

    cosine_similarities3 = all_cosine_sims[0]

    # create an array that is formatted so that it can be passed into the sklearn model pipeline
    elx = np.recarray(shape=(len(essay_list),),
                      dtype=[('text', object), ('length', float),
                             ('similarity3', float)])

    elx['text'][:] = essay_list[:]
    elx['length'][:] = wordcount[:]
    elx['similarity3'][:] = cosine_similarities3[:]

    # still experimenting with what degree polynomialfeature to apply to sentence length
    # for degree in range(11):
    correct_list = []
    pred_list = []
    pred_prob_list = []

    # tfidfvectorizer vs countvectorizer , some problems work better than others
    # tfidf has some problems it does better on, but the kappa score sometimes is very bad, so count probably is better

    txt_len_features3 = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length'),
            PolynomialFeatures(degree=1)
        )),
        ('text_features', make_pipeline(
            ItemSelector(key='text'),
            CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize, ngram_range=(1, 1))
        )),
        ('text_features_2', make_pipeline(
            ItemSelector(key='similarity3'),
            PolynomialFeatures(degree=1)
        ))
    ], transformer_weights={"length_features": 1, "text_features": 1, "text_features_2": 10})

    classifier_pipe3 = make_pipeline(txt_len_features3, RidgeClassifier())

    # In this setup, the first essay/score is the one we are testing
    trainE = elx[1:]
    train_scores = label[1:]
    testE = elx[0].reshape((1,))
    test_score = label[0]

    splits = 3
    # default CV split is 3: if
    for s in set(train_scores):
        splits = min(train_scores.count(s), splits)
    splits = max(splits, 2)

    models = [
        classifier_pipe3
    ]

    result_prob_top = float('-inf')
    result = 0
    for model in models:
        # train
        model.fit(trainE, train_scores)

        # get results
        result_temp = model.predict(testE)
        result_probs_temp = model.decision_function(testE)

        # compare decision functions
        if np.max(result_probs_temp[0]) > result_prob_top:
            result = result_temp
            result_prob_top = np.max(result_probs_temp[0])

    if test_score == result[0]:
        return 1
    else:
        return 0

# for multithreading purposes, lock will help to ensure safe read/write to file for shared results
def init_child(lock_):
    global lock
    lock = lock_

def essay_trial_acc(essay_list, labels, essay_index, train_size):

    accs = []
    essay_list_noholdout = essay_list[:essay_index]
    essay_list_noholdout.extend(essay_list[essay_index+1:])
    label_noholdout = labels[:essay_index]
    label_noholdout.extend(labels[essay_index+1:])
    for trial in range(20):
        resample_essay, resample_label = resample(essay_list_noholdout, label_noholdout, n_samples=train_size, replace=False)
        essay_list_subset = [essay_list[essay_index]]
        essay_list_subset.extend(resample_essay)
        label_subset = [labels[essay_index]]
        label_subset.extend(resample_label)
        accs.append(run_model_on_problem(essay_list_subset, label_subset))

    return [essay_list[essay_index], labels[essay_index], train_size, np.mean(accs)]

if __name__ == '__main__':

    assignment_pidlist = data_utils.load_pidlist(training_path)

    for pid in assignment_pidlist:
        folder_name = '{}_trainingsize_engageny'.format(pid)
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "only3sim_trainingsize_runs2", folder_name))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            #skip over problems that are already done
            continue

        #training_path = 'Final_Filtered_Data_John.csv'#'filter_engageny_openresponse.csv'  # name of dataset
        essay_list, label, problem_id, count_one, question_list = data_utils.load_open_response_data(training_path, pid)

        # ignore problems with only 1 grade category
        if len(set(label)) <= 1:
            continue

        df = pd.DataFrame(columns=['essay', 'label', 'n_data', '30trial_accuracy'])

        lock = Lock()

        #training_sizes = [int(n) for n in range(5, len(essay_list))]
        training_sizes = [5, 10, 15, 20, 25, 30, 35, 40]

        for training_size in training_sizes:
            essay_indexes = [int(i) for i in range(len(essay_list))]
            results = []
            # go through each problem of interest, use multithreading
            with closing(Pool(3, initializer=init_child, initargs=(lock,))) as pool:
                results = (pool.starmap(essay_trial_acc, zip(itertools.repeat(essay_list), itertools.repeat(label),
                                                            essay_indexes, itertools.repeat(training_size))))
            for res in results:
                df = df.append({
                    'essay': res[0],
                    'label': res[1],
                    'n_data': res[2],
                    '30trial_accuracy': res[3]
                }, ignore_index=True)
            print("n_data:{}, accuracy: {}\n".format(training_size, np.mean(np.array(results)[:, 3])))
            with open(out_dir + '/params', 'a') as f:
                f.write("n_data:{}, accuracy: {}\n".format(training_size, np.mean(np.array(results)[:, 3])))
            df.to_csv(out_dir + "/" + 'holdout_test_results.csv')