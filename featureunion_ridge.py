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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import cohen_kappa_score
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
from multiprocessing import Pool, Lock
from contextlib import closing
import itertools
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats import entropy

# disable tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_sample = 10

# print flags info
orig_stdout = sys.stdout
timestamp = str(int(time.time()))

# name of file to load response data
training_path = 'Final_Filtered_Data_John_update.csv'  # 'Final_Filtered_Data_John.csv'#

# name of folder in which to store results
top_folder_name = 'jesus2_timestamp{}'.format(timestamp)

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
            try:
                vectorizer_temp = TfidfVectorizer(tokenizer=TreebankWordTokenizer().tokenize, analyzer='char',
                                                  ngram_range=(2, n))

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
def run_model_on_problem(pid, folder_name, training_path):
    pid = pid
    folder_name_pid = '{}/individual_problems/{}'.format(folder_name, pid)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", folder_name_pid))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(pid)
        return

    df_temp = pd.DataFrame(
        columns=['PID', 'Problem Count', 'Mean', 'Variance', '0', '0.25', '0.5', '0.75', '1', 'LOO Accuracy',
                 'Cohen Kappa', 'Number Confident (>0)', 'Accuracy of Confident', 'Number High Confident (>0.5)',
                 'Accuracy High Confident', 'Precision', 'Recall', 'Fscore', 'Support'])

    df_details_temp = pd.DataFrame(columns=['PID', 'Label', 'Predicted', 'PredictionCorrect',
                                       'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'Text'])

    ind_model_dfs = [df_details_temp, df_details_temp, df_details_temp, df_details_temp]

    df_params_temp = pd.DataFrame(columns=['PID', 'weights', 'ngrams', 'degree', 'choice'])

    lock.acquire()
    essay_list, label, problem_id, count_one, question_list = data_utils.load_open_response_data(training_path, pid)
    lock.release()

    # skip over problems that only have 1 distinct score
    if len(set(label)) == 1:
        return


    n_total = len(essay_list)

    sent_size_list = np.array([len(essay) for essay in essay_list])
    max_sent_size = max(sent_size_list)
    mean_sent_size = int(np.mean(list(map(len, [essay for essay in essay_list]))))

    question_sent_size_list = np.array(
        [len(question) for question in question_list])
    question_max_sent_size = max(question_sent_size_list)
    question_mean_sent_size = int(np.mean(list(map(len, [question for question in question_list]))))

    print('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))

    with open(out_dir + '/params', 'a') as f:
        f.write('problem_id: {}\n'.format(pid))
        f.write('number of problems: {}\n'.format(len(essay_list)))
        f.write('max sentence size: {} \nmean sentence size: {}\nquestion max sentence size: {}\n'
                'question mean sentence size:{}\n'.format(max_sent_size, mean_sent_size, question_max_sent_size,
                                                          question_mean_sent_size))
        f.write(
            'score distribution -- \n0.0: {}\n0.25: {}\n 0.5: {}\n0.75: {}\n1.0: {}\nmean: {}\nvariance: {}\n'.format(
                label.count(0), label.count(1), label.count(2), label.count(3), label.count(4),
                np.mean(np.array(label) / 4), np.var(np.array(label) / 4)
            ))

    # data size
    n_train = n_total - 1
    n_test = 1

    print('The size of training data: {}'.format(n_train))
    print('The size of testing data: {}'.format(n_test))

    # tf input
    n_steps = max_sent_size
    print("n_steps = max_sent_size = ")
    print(n_steps)
    print("question_max_sent_size = ")
    print(str(question_max_sent_size))

    # get word count for each student response
    wordcount = []
    for i in range(len(essay_list)):
        wordcount.append(len(essay_list[i]))
        essay_list[i] = ' '.join(word for word in essay_list[i])
    wordcount = np.array(wordcount)

    # get cosine similarity features
    # all_cosine_sims = [
    #     get_cosine_sims(essay_list, label, 3),
    #     get_cosine_sims(essay_list, label, 7),
    #     get_cosine_sims(essay_list, label, 11)
    # ]
    # cosine_similarities3 = all_cosine_sims[0]
    # cosine_similarities7 = all_cosine_sims[1]
    # cosine_similarities11 = all_cosine_sims[2]


    # create an array that is formatted so that it can be passed into the sklearn model pipeline
    elx = np.recarray(shape=(len(essay_list),),
                      dtype=[('text', object), ('length', float),
                             ('similarity3', float),
                             ('similarity7', float),
                             ('similarity11', float)])

    elx['text'][:] = essay_list[:]
    elx['length'][:] = wordcount[:]
    # elx['similarity3'][:] = cosine_similarities3[:]
    # elx['similarity7'][:] = cosine_similarities7[:]
    # elx['similarity11'][:] = cosine_similarities11[:]

    correct_list = []
    pred_list = []
    pred_prob_list = []

    ##### FeatureUnions to be used in models #####

    txt_len_features3 = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length'),
            PolynomialFeatures(degree=1)
        )),
        ('text_features', make_pipeline(
            ItemSelector(key='text'),
            CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
        )),
       # ('text_features_2', make_pipeline(
       #     ItemSelector(key='similarity3'),
       #     PolynomialFeatures(degree=1)
       # ))
    ], transformer_weights={"length_features": 1, "text_features": 1})#, "text_features_2": 10

    txt_len_features7 = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length'),
            PolynomialFeatures()
        )),
        ('text_features', make_pipeline(
            ItemSelector(key='text'),
           CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
        )),
        ('text_features_2', make_pipeline(
            ItemSelector(key='similarity7'),
            PolynomialFeatures()
        ))
    ])

    txt_len_features11 = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length'),
            PolynomialFeatures()
        )),
        ('text_features', make_pipeline(
            ItemSelector(key='text'),
            CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
        )),
        ('text_features_2', make_pipeline(
            ItemSelector(key='similarity11'),
            PolynomialFeatures()
        ))
    ])

    txt_len_features = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length')
        )),
        ('text_features', make_pipeline(
            ItemSelector(key='text'),
            CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
        ))
    ])

    len_features = FeatureUnion([
        ('length_features', make_pipeline(
            ItemSelector(key='length'),
            PolynomialFeatures()
        ))
    ])

    ##### end of FeatureUnions to be used in models #####

    # go through each essay using leave-one-out
    for i in range(len(essay_list)):
        # select and train on all but one student answer
        trainE = np.concatenate((elx[:i], elx[i + 1:]))
        train_scores = np.concatenate((label[:i], label[i + 1:]))

        trainE, train_scores = shuffle(trainE, train_scores)

        classifier_pipe3 = make_pipeline(txt_len_features3, DecisionTreeClassifier(max_depth=3))#max_depth=2)) #RidgeClassifier())
        classifier_pipe7 = make_pipeline(txt_len_features7, RidgeClassifier())
        classifier_pipe11 = make_pipeline(txt_len_features11, RidgeClassifier())
        classifier_pipe = make_pipeline(txt_len_features, RidgeClassifier())
        classifier_pipe_len = make_pipeline(len_features, RidgeClassifier())


        param_grid_3features = {'featureunion__transformer_weights': [
            {"length_features": 1, "text_features": 0, "text_features_2": 10},
            {"length_features": 1, "text_features": 1, "text_features_2": 10},
            # extra parameters, some work marginally better for certain problems. not really worth the extra computation
            #  {"length_features": 1, "text_features": 0.5, "text_features_2": 10},
            #  {"length_features": 5, "text_features": 0, "text_features_2": 1},
            #  {"length_features": 5, "text_features": 1, "text_features_2": 1},
            #  {"length_features": 5, "text_features": 0.5, "text_features_2": 1}

        ],
            'featureunion__text_features__countvectorizer__ngram_range': [(1, 1)],
            'featureunion__length_features__polynomialfeatures__degree': [1],
            'featureunion__text_features_2__polynomialfeatures__degree': [1]
        }

        param_grid_2features = {'featureunion__transformer_weights': [
            {"poly_features": 1, "text_features": 0},
            {"poly_features": 1, "text_features": 1},
            {"poly_features": 0.3, "text_features": 1},
            {"poly_features": 1, "text_features": 0.5}],
            'featureunion__text_features__countvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]
        }

        param_grid_1features = {
            'featureunion__length_features__polynomialfeatures__degree': [1, 2, 3, 4]
        }

        # train multiple models and choose the one that has the greatest decision function to choose predictions
        models = [
            # currently only using the basic ridge classifier with no gridsearch
            classifier_pipe3
            # uncomment to use multiple models, which will be comapred based on the decision_function's output
            #GridSearchCV(classifier_pipe3, param_grid_3features),
            #GridSearchCV(classifier_pipe7, param_grid_3features),
            #GridSearchCV(classifier_pipe11, param_grid_3features),
            #GridSearchCV(classifier_pipe, param_grid_2features),
            #GridSearchCV(classifier_pipe_len, param_grid_1features)
        ]

        result_prob_top = float('-inf')
        result_probs = []
        result = 0
        chosen = -1
        for model in models:
            # train
            model.fit(trainE, train_scores)

            # get results
            result_temp = model.predict(elx[i].reshape((1,)))
            result_probs_temp = model.predict_proba(elx[i].reshape((1,)))

            #for looking at the weights and such of the trained model, for the classifier_pipe3
            #an extra step is probably needed before the named_steps[...] if gridsearch models are used
            # print(model.named_steps['featureunion'].transformer_list[1][1].named_steps['countvectorizer'].vocabulary_)
            # print(model.named_steps['featureunion'].transform(elx[i].reshape((1,))))
            # print(model.named_steps['decisiontreeclassifier'].feature_importances_)
            # export_graphviz(model.named_steps['decisiontreeclassifier'], out_file='tree_maxfeats.dot')
            # print(model.named_steps['ridgeclassifier'].intercept_[0])
            # quit()

            # compare decision functions
            if np.max(result_probs_temp[0]) > result_prob_top:
                result_probs = result_probs_temp
                result = result_temp
                chosen = models.index(model)
                result_prob_top = np.max(result_probs_temp[0])

            # save details about the decision function for each model
            temp_DFs = ['', '', '', '', '']
            if isinstance(result_probs_temp[0], float):
                temp_DFs[0] = result_probs_temp[0]
            else:
                for res_i in range(len(result_probs_temp[0])):
                    temp_DFs[res_i] = result_probs_temp[0][res_i]
            ind_model_dfs[models.index(model)] = ind_model_dfs[models.index(model)].append({
                'PID': pid,
                'Label': label[i],
                'Predicted': result_temp,
                'PredictionCorrect': label[i] == result_temp,
                'DF1': 0,
                'DF2': 0,
                'DF3': 0,
                'DF4': 0,
                'DF5': 0,
                'Text': essay_list[i]
            }, ignore_index=True)
            ind_model_dfs[models.index(model)].at[ind_model_dfs[models.index(model)].shape[0] - 1, 'DF1'] = temp_DFs[0]
            ind_model_dfs[models.index(model)].at[ind_model_dfs[models.index(model)].shape[0] - 1, 'DF2'] = temp_DFs[1]
            ind_model_dfs[models.index(model)].at[ind_model_dfs[models.index(model)].shape[0] - 1, 'DF3'] = temp_DFs[2]
            ind_model_dfs[models.index(model)].at[ind_model_dfs[models.index(model)].shape[0] - 1, 'DF4'] = temp_DFs[3]
            ind_model_dfs[models.index(model)].at[ind_model_dfs[models.index(model)].shape[0] - 1, 'DF5'] = temp_DFs[4]

        if label[i] == result[0]:
            correct_list.append(1)
        else:
            correct_list.append(0)
        pred_list.append(result[0])
        pred_prob_list.append(np.max(result_probs[0]))

        DFs = ['', '', '', '', '']
        if isinstance(result_probs[0], float):
            DFs[0] = result_probs[0]
        else:
            for res_i in range(len(result_probs[0])):
                DFs[res_i] = result_probs[0][res_i]

        df_details_temp = df_details_temp.append({
            'PID': pid,
            'Label': label[i],
            'Predicted': result[0],
            'PredictionCorrect': correct_list[i],
            'DF1': 0,
            'DF2': 0,
            'DF3': 0,
            'DF4': 0,
            'DF5': 0,
            'Text': essay_list[i]
        }, ignore_index=True)

        df_details_temp.at[df_details_temp.shape[0] - 1, 'DF1'] = DFs[0]
        df_details_temp.at[df_details_temp.shape[0] - 1, 'DF2'] = DFs[1]
        df_details_temp.at[df_details_temp.shape[0] - 1, 'DF3'] = DFs[2]
        df_details_temp.at[df_details_temp.shape[0] - 1, 'DF4'] = DFs[3]
        df_details_temp.at[df_details_temp.shape[0] - 1, 'DF5'] = DFs[4]

        # save info about the chosen gridsearch parameters / model
        # used for checking what parameters ended up being used by the gridsearch models
        df_params_temp = df_params_temp.append({
            'PID': pid,
            'weights': 0,#models[chosen].best_params_['featureunion__transformer_weights'],
            'ngrams': 0,#models[chosen].best_params_['featureunion__text_features__countvectorizer__ngram_range'],
            'degree': 0,#models[chosen].best_params_['featureunion__length_features__polynomialfeatures__degree'],
            'choice': chosen
        }, ignore_index=True)
        del (models)

    print("LOO CV Accuracy: {}".format(np.mean(correct_list)))
    kap = 0#cohen_kappa_score(pred_list, label)
    print("Cohen Kappa Score: {}".format(kap))
    with open(out_dir + '/params', 'a') as f:
        # f.write("Poly {} LOO CV Accuracy: {}\n".format(degree, np.mean(correct_list)))
        f.write("LOO CV Accuracy: {}\n".format(np.mean(correct_list)))
        f.write("Cohen Kappa: {}\n".format(kap))
    confident_correct_list = []
    high_confident_correct_list = []
    for k in range(len(correct_list)):
        if pred_prob_list[k] > 0:
            confident_correct_list.append(correct_list[k])
        if pred_prob_list[k] > 0.5:
            high_confident_correct_list.append(correct_list[k])
    print("Confident count: {}".format(len(confident_correct_list)))
    print("Confident accuracy: {}".format(np.mean(confident_correct_list)))
    print("High Confident count: {}".format(len(high_confident_correct_list)))
    print("High Confident accuracy: {}".format(np.mean(high_confident_correct_list)))

    with open(out_dir + '/params', 'a') as f:
        f.write("Confident count: {}\n".format(len(confident_correct_list)))
        f.write("Confident accuracy: {}\n".format(np.mean(confident_correct_list)))
        f.write("High Confident count: {}\n".format(len(high_confident_correct_list)))
        f.write("High Confident accuracy: {}\n".format(np.mean(high_confident_correct_list)))


    # save information about the run on this specific problemset.
    # some of the info that's being saved is redundant, but might come in handy
    ind = 0
    for ind_df in ind_model_dfs:
        ind_df.to_csv(out_dir+'/'+str(pid)+'_ind_test_details_'+str(ind)+'.csv', index=False)
        ind += 1
    df_params_temp.to_csv(out_dir + '/' + 'test_params'+str(pid)+'.csv', index=False)
    df_details_temp.to_csv(out_dir + '/' + 'test_params'+str(pid)+'.csv', index=False)

    # save dataframe stuff
    lock.acquire()
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", folder_name))

    df = pd.read_csv(out_dir + '/' + 'test_results.csv')
    df_params = pd.read_csv(out_dir + '/' + 'test_params.csv')
    df_details = pd.read_csv(out_dir + '/' + 'test_details.csv')

    df = pd.concat([df, (pd.DataFrame(np.column_stack(
        [pid, len(essay_list),
         label.count(0), label.count(1), label.count(2), label.count(3), label.count(4),
         np.mean(correct_list), kap, len(confident_correct_list), np.mean(confident_correct_list),
         len(high_confident_correct_list), np.mean(high_confident_correct_list), 0, 0, 0, 0]),
        columns=['PID', 'Problem Count', '0', '0.25', '0.5', '0.75', '1', 'LOO Accuracy',
                 'Cohen Kappa', 'Number Confident (>0)', 'Accuracy of Confident', 'Number High Confident (>0.5)',
                 'Accuracy High Confident', 'Precision', 'Recall', 'Fscore', 'Support']))], ignore_index=True,
                   sort=False)
    df_params = df_params.append(df_params_temp, ignore_index=True, sort=False)
    df_details = df_details.append(df_details_temp, ignore_index=True, sort=False)

    df_params.to_csv(out_dir + '/' + 'test_params.csv', index=False)
    df.to_csv(out_dir + '/' + 'test_results.csv', index=False)
    df_details.to_csv(out_dir + '/' + 'test_details.csv', index=False)
    lock.release()
    return

# for multithreading purposes, lock will help to ensure safe read/write to file for shared results
def init_child(lock_):
    global lock
    lock = lock_

if __name__ == '__main__':
    assignment_pidlist = data_utils.load_pidlist(training_path)

    df = pd.DataFrame(
        columns=['PID', 'Problem Count',  '0', '0.25', '0.5', '0.75', '1', 'LOO Accuracy',
                 'Cohen Kappa', 'Number Confident (>0)', 'Accuracy of Confident', 'Number High Confident (>0.5)',
                 'Accuracy High Confident', 'Precision', 'Recall', 'Fscore', 'Support'])

    df_details = pd.DataFrame(columns=['PID', 'Label', 'Predicted', 'PredictionCorrect',
                                       'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'Text'])

    df_params = pd.DataFrame(columns=['PID', 'weights', 'ngrams', 'degree', 'choice'])

    # if restarting from the middle of a previously abandoned run...
    # df = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_results.csv")
    # df_params = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_params.csv")
    # df_details = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_details.csv")

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", top_folder_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(out_dir + '/' + 'test_results.csv', index=False)
    df_details.to_csv(out_dir + '/' + 'test_details.csv', index=False)
    df_params.to_csv(out_dir + '/' + 'test_params.csv', index=False)

    lock = Lock()

    # go through each problem of interest, use multithreading
    with closing(Pool(3, initializer=init_child, initargs=(lock,))) as pool:
        pool.starmap(run_model_on_problem, zip(assignment_pidlist, itertools.repeat(top_folder_name),
                                               itertools.repeat(training_path)))
