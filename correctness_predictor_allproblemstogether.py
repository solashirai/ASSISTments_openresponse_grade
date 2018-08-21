from __future__ import division, print_function, absolute_import
import os
import sys
import time
#test
import numpy as np
import data_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.linear_model import RidgeClassifier, BayesianRidge, SGDClassifier, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import cohen_kappa_score
from nltk.tokenize import TreebankWordTokenizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from multiprocessing import Pool, Lock
from contextlib import closing
import itertools
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

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

# data_path = 'main_baseline_results/newfiltertests/whichmodel_byaccuracy.csv'  # name of dataset
# data_df = pd.read_csv(data_path)
# pidlist = data_df['problem_id']
# best_model = data_df['whichlab']

detail_dir = 'goodruns/1533297156_newestfilter_trial2/'

def get_problem_stats(pid):
    training_path = detail_dir + 'test_details.csv'  # name of dataset
    lock.acquire()
    essay_list, label, problem_id, predicted_label, true_label = data_utils.load_detail_data(training_path, pid)
    lock.release()

    if len(essay_list) == 0:
        return [], [[],[],[]]

    sent_size_list = np.array([len(essay) for essay in essay_list])
    max_sent_size = max(sent_size_list)
    min_sent_size = min(sent_size_list)
    range_sent_size = max_sent_size - min_sent_size
    mean_sent_size = int(np.mean(list(map(len, [essay for essay in essay_list]))))
    variance_sent_size = np.var(sent_size_list)

    n_count = len(essay_list)
    n_classes = len(set(true_label))

    score_counts_unnormalized = [0, 0, 0, 0, 0]
    score_counts = []
    for s in range(5):
        score_counts_unnormalized[s] = true_label.count(s)
        score_counts.append(true_label.count(s) / len(true_label))
    mean_score = np.mean(true_label)
    variance_score = np.var(true_label)
    majority_score_percentage = np.max(score_counts)
    range_score_counts = np.max(score_counts) - np.min(score_counts)
    variance_score_counts = np.var(score_counts * n_count)

    string_essay_list = []
    for essay in essay_list:
        string_essay_list.append(' '.join(essay))

    vectorizer = TfidfVectorizer(
        tokenizer=TreebankWordTokenizer().tokenize)  # , binary=True) #binary or not seems to have little to no effect?
    vectorized_essays = vectorizer.fit_transform(string_essay_list)
    vocab_size = len(vectorizer.vocabulary_)

    # compute essay similarity and get the average
    essay_similarities = (vectorized_essays * vectorized_essays.T).A
    unique_similarities = []
    for e in range(len(essay_similarities)):
        unique_similarities += list(essay_similarities[e][e + 1:])
    mean_sent_similarity = np.mean(unique_similarities)
    median_sent_similarity = np.median(unique_similarities)
    range_sent_similarity = np.max(unique_similarities) - np.min(unique_similarities)
    variance_sent_similarity = np.var(unique_similarities)

    # for computing essay similarity for specified scores
    vectorized_essays_by_score = [[], [], [], [], []]
    vectorized_essays_by_not_score = [[], [], [], [], []]
    unique_similarities_by_score = [[], [], [], [], []]
    max_similarity_by_score = []
    max_similarity_to_others_by_score = []

    essay_length_by_score = [[], [], [], [], []]
    essay_length_by_not_score = [[], [], [], [], []]
    length_similarity_by_score = []
    length_similarity_to_others_by_score = []
    non_major_score_similarity_sum = 0
    fullscore_mean_similarities = []
    fullscore_unique_vocab_count = 0

    for s in range(len(essay_list)):
        vectorized_essays_by_score[true_label[s]].append(string_essay_list[s])
        essay_length_by_score[true_label[s]].append(sent_size_list[s])
    for x in range(len(vectorized_essays_by_score)):
        for y in range(len(vectorized_essays_by_score)):
            if x != y:
                vectorized_essays_by_not_score[x].extend(vectorized_essays_by_score[y])
                essay_length_by_not_score[x].extend(essay_length_by_score[y])
    min_similarity_by_score = 1

    for v in range(len(vectorized_essays_by_score)):
        if vectorized_essays_by_score[v]:
            if v == 4:
                if len(vectorized_essays_by_score[v]) <= 1:
                    fullscore_mean_similarities = 0
                    fullscore_unique_vocab_count = 0
                else:
                    for u in range(len(vectorized_essays_by_score[v])):
                        fullscore_vectorizer = TfidfVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
                                                               analyzer='char', ngram_range=(2, 7))
                        fullscore_sentences_subset = np.append(vectorized_essays_by_score[v][:u],
                                                               vectorized_essays_by_score[v][u + 1:])
                        fullscore_vectorizer.fit(fullscore_sentences_subset)
                        fullscore_sentences_vectorized = fullscore_vectorizer.transform(vectorized_essays_by_score[v])
                        top_vecs = np.zeros(fullscore_sentences_vectorized[0].shape)
                        for w in range(len(vectorized_essays_by_score[v])):
                            if w != u:
                                top_vecs += fullscore_sentences_vectorized[w]
                        top_vecs = top_vecs / (len(vectorized_essays_by_score[v]) - 1)
                        fullscore_mean_similarities.append((fullscore_sentences_vectorized[u] * top_vecs.T).A[0][0])

                        extra_vectorizer = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
                        extra_vectorizer.fit(fullscore_sentences_subset)
                        for word in CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize).tokenizer(
                                vectorized_essays_by_score[v][u]):
                            if word not in extra_vectorizer.vocabulary_:
                                fullscore_unique_vocab_count += 1
            if len(vectorized_essays_by_score[v]) == 1:
                length_similarity_by_score.append(0)
            else:
                vectorized_essays_by_score[v] = vectorizer.transform(vectorized_essays_by_score[v])
                essay_similarities_by_score = (vectorized_essays_by_score[v] * vectorized_essays_by_score[v].T).A
                max_similarities = []
                for u in range(len(essay_similarities_by_score)):
                    unique_similarities_by_score[v] += list(essay_similarities_by_score[u][u + 1:])
                    if len(unique_similarities_by_score[v]) <= 1:
                        maxsim = 0
                    else:
                        maxsim = np.max(np.append(essay_similarities_by_score[u][:u],
                                                  essay_similarities_by_score[u][u + 1:]))
                    max_similarities.append(maxsim)
                max_similarity_by_score.append(np.mean(max_similarities))
                mean_usim = np.mean(unique_similarities_by_score[v])
                min_similarity_by_score = min(mean_usim, min_similarity_by_score)
                if score_counts[v] != max(score_counts):
                    non_major_score_similarity_sum += mean_usim * score_counts[v]

                # deal with analyzing essay similarity of essays in one score category with those from scores
                max_similarities_to_other = []
                vectorized_essays_by_not_score[v] = vectorizer.transform(vectorized_essays_by_not_score[v])
                essay_similarities_by_score_to_other = (
                vectorized_essays_by_score[v] * vectorized_essays_by_not_score[v].T).A
                for u in range(len(essay_similarities_by_score_to_other)):
                    max_similarities_to_other.append(np.max(essay_similarities_by_score_to_other[u]))
                max_similarity_to_others_by_score.append(
                    np.mean(essay_similarities_by_score_to_other) * score_counts[v])

                length_similarities = []
                length_similarities_to_other = []
                for u in range(len(essay_length_by_score[v])):
                    dists = []
                    for x in range(len(essay_length_by_score[v])):
                        if x != u:
                            dists.append((essay_length_by_score[v][u] - essay_length_by_score[v][x]) ** 2)
                    length_similarities.append(np.mean(dists))  # ** 0.5)
                    other_dists = []
                    for x in range(len(essay_length_by_not_score[v])):
                        other_dists.append((essay_length_by_score[v][u] - essay_length_by_not_score[v][x]) ** 2)
                    length_similarities_to_other.append(np.mean(other_dists))  # ** 0.5)
                length_similarity_by_score.append(np.mean(length_similarities) * score_counts[v])
                length_similarity_to_others_by_score.append(
                    np.mean(length_similarities_to_other) * score_counts[v] * n_count)
                # max_similarity_to_others_by_score.append(np.sum(np.where(np.array(max_similarities_to_other) > np.array(max_similarities), 1, 0)))
        else:
            vectorized_essays_by_score[v] = np.array([])
            if v == 4:
                fullscore_mean_similarities.append(0)

    kap = cohen_kappa_score(predicted_label, true_label)
    acc = accuracy_score(predicted_label, true_label)

    length_similarity_by_score_withzeros = [0, 0, 0, 0, 0]
    ll = 0
    for l in range(5):
        if essay_length_by_score[l] != []:
            length_similarity_by_score_withzeros[l] = length_similarity_by_score[ll]
            ll += 1

    return [n_count,
            n_classes,
            majority_score_percentage * n_count,
            variance_score_counts,
            np.mean(max_similarity_by_score),
            non_major_score_similarity_sum,
            variance_sent_similarity,
            np.mean(max_similarity_to_others_by_score),
           # np.mean(length_similarity_by_score),
           # np.mean(length_similarity_to_others_by_score),
            np.mean(fullscore_mean_similarities),# * majority_score_percentage * n_count,
            vocab_size,
            fullscore_unique_vocab_count,
            # score_counts_unnormalized[0],
            # score_counts_unnormalized[1],
            # score_counts_unnormalized[2],
            # score_counts_unnormalized[3],
            # score_counts_unnormalized[4],
            # length_similarity_by_score_withzeros[0],
            # length_similarity_by_score_withzeros[1],
            # length_similarity_by_score_withzeros[2],
            # length_similarity_by_score_withzeros[3],
            # length_similarity_by_score_withzeros[4]
            ], [essay_list, predicted_label, true_label]

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key == 'text' or self.key == 'stats':
            return data_dict[self.key]
        else:
            return data_dict[self.key].reshape(-1, 1)

class FakeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.a = ''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # not sure why, but need to fix shape like this and make into np array
        X2 = []
        for xx in X:
            X2.append(xx)
        return np.array(X2)

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

            vectorizer_temp.fit(fullscore_essays)

            vec_essays_temp = vectorizer_temp.transform(essay_list)
            top_vecs_temp = np.zeros(vec_essays_temp[0].shape)
            for fullscore_i in fullscore_indices:
                top_vecs_temp += vec_essays_temp[fullscore_i]
            top_vecs_temp = top_vecs_temp / len(fullscore_indices)
            cosine_similarities_temp.append((vec_essays_temp[x] * top_vecs_temp.T).A[0][0])
    return cosine_similarities_temp

# mostly just set up as a separate function to allow mutithreading
def run_model_on_problem(pid, folder_name, training_path):
    folder_name_pid = '{}/individual_problems/{}'.format(folder_name, pid)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, detail_dir+'/correctness_predictor_allproblemstogether_run/', folder_name_pid))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(pid)
        return

    df_temp = pd.DataFrame(
        columns=['PID', 'Problem Count', 'Mean', 'Variance', '0', '1', 'LOO Accuracy',
                 'Cohen Kappa', 'Number Confident (>0)', 'Accuracy of Confident', 'Number High Confident (>0.5)',
                 'Accuracy High Confident', 'Precision', 'Recall', 'Fscore', 'Support'])

    df_details_temp = pd.DataFrame(columns=['PID', 'Label', 'Predicted', 'PredictionCorrect',
                                       'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'Text'])

    ind_model_dfs = [df_details_temp, df_details_temp, df_details_temp, df_details_temp]

    problem_stats, essay_details = get_problem_stats(pid)
    essay_list = essay_details[0]
    p_label = essay_details[1]
    t_label = essay_details[2]
    label = np.where(np.array(p_label) == np.array(t_label), 1, 0).tolist()

    # skip over problems that only have 1 distinct score
    if len(set(label)) <= 1:
        return

    # this should all be handled properly now
    n_total = len(essay_list)

    sent_size_list = np.array([len(essay) for essay in essay_list])
    max_sent_size = max(sent_size_list)
    mean_sent_size = int(np.mean(list(map(len, [essay for essay in essay_list]))))

    print('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))

    with open(out_dir + '/params', 'a') as f:
        f.write('problem_id: {}\n'.format(pid))
        f.write('number of problems: {}\n'.format(len(essay_list)))
        f.write(
            'score distribution -- \n0.0: {}\n1: {}\n mean: {}\nvariance: {}\n'.format(
                label.count(0), label.count(1),
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

    # get word count for each student response
    wordcount = []
    for i in range(len(essay_list)):
        wordcount.append(len(essay_list[i]))
        essay_list[i] = ' '.join(word for word in essay_list[i])
    wordcount = np.array(wordcount)
    # wordcount = wordcount / np.max(wordcount)

    all_cosine_sims = [
        get_cosine_sims(essay_list, label, 3),
        get_cosine_sims(essay_list, label, 7),
        get_cosine_sims(essay_list, label, 11)
    ]

    cosine_similarities3 = all_cosine_sims[0]
    cosine_similarities7 = all_cosine_sims[1]
    cosine_similarities11 = all_cosine_sims[2]
    problem_stats = [problem_stats,]*n_total

    # create an array that is formatted so that it can be passed into the sklearn model pipeline
    # elx = np.recarray(shape=(len(essay_list),),
    #                   dtype=[('length', float), #('text', object),
    #                          ('similarity3', float),
    #                          ('similarity7', float),
    #                          ('similarity11', float),
    #                          ('stats', object)])
    #
    # #elx['text'][:] = essay_list[:]
    # elx['length'][:] = wordcount[:]
    # elx['similarity3'][:] = cosine_similarities3[:]
    # elx['similarity7'][:] = cosine_similarities7[:]
    # elx['similarity11'][:] = cosine_similarities11[:]
    # elx['stats'][:] = problem_stats[:]

    return wordcount.tolist(), cosine_similarities7, problem_stats, label, essay_list

# for multithreading purposes, lock will help to ensure safe read/write to file for shared results
def init_child(lock_):
    global lock
    lock = lock_

if __name__ == '__main__':
    # name of file to load response data
    training_path = 'filter_engageny_openresponse.csv'
    assignment_pidlist = data_utils.load_pidlist(training_path)
    # assignment_pidlist = assignment_pidlist[:6]

    df = pd.DataFrame(
        columns=['PID', 'Problem Count',  '0', '1', 'LOO Accuracy',
                 'Cohen Kappa', 'Number Confident (>0)', 'Accuracy of Confident', 'Number High Confident (>0.5)',
                 'Accuracy High Confident', 'Precision', 'Recall', 'Fscore', 'Support'])

    df_details = pd.DataFrame(columns=['PID', 'Label', 'Predicted', 'PredictionCorrect',
                                       'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'Text'])

    # if restarting from the middle of a previously abandoned run...
    # df = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_results.csv")
    # df_params = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_params.csv")
    # df_details = pd.read_csv("runs/hugetest_modifieddfcomparison_engageny/test_details.csv")

    folder_name = '{}'.format(timestamp)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, detail_dir + "correctness_predictor_allproblemstogether_run/", folder_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df.to_csv(out_dir + '/' + 'test_results.csv', index=False)
    df_details.to_csv(out_dir + '/' + 'test_details.csv', index=False)

    lock = Lock()

    # go through each problem of interest, use multithreading
    with closing(Pool(3, initializer=init_child, initargs=(lock,))) as pool:
        problem_stat_data = pool.starmap(run_model_on_problem, zip(assignment_pidlist, itertools.repeat(folder_name),
                                               itertools.repeat(training_path)))

    wordcounts = []
    cosine_similarities7 = []
    problem_stats = []
    label = []
    essay_list = []
    for data in problem_stat_data:
        if data:
            wordcounts.extend(data[0])
            cosine_similarities7.extend(data[1])
            problem_stats.extend(data[2])
            label.extend(data[3])
            essay_list.extend(data[4])

    elx = np.recarray(shape=(len(label),),
                      dtype=[('text', object),('length', float),
                             ('similarity7', float),
                             ('stats', object)])

    elx['text'][:] = essay_list[:]
    elx['length'][:] = wordcounts[:]
    elx['similarity7'][:] = cosine_similarities7[:]
    elx['stats'][:] = problem_stats[:]

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
            )),
            ('problem_stats', make_pipeline(
                ItemSelector(key='stats'),
                FakeVectorizer(),
                PolynomialFeatures()
            ))])

    param_grid_3features = {
                'featureunion__text_features__countvectorizer__ngram_range': [(1, 1)],
                'featureunion__length_features__polynomialfeatures__degree': [1],
                'featureunion__text_features_2__polynomialfeatures__degree': [1],
                'featureunion__problem_stats__polynomialfeatures__degree': [0, 1, 2]
            }

    accuracies = []
    kappas = []
    for trial in range(5):
        trainE, testE, trainScores, testScores = train_test_split(elx, label,  # problem_kappa,
                                                                  test_size=0.2, stratify=label)  # , stratify=problem_kappa)

        classifier_pipe7 = make_pipeline(txt_len_features7, DecisionTreeClassifier(max_features='sqrt', class_weight='balanced'))# RidgeClassifier(normalize=True, class_weight='balanced')) #RandomForestClassifier())#
        model = GridSearchCV(classifier_pipe7, param_grid_3features, n_jobs=3)

        model.fit(trainE, trainScores)

        training_pred = model.predict(trainE)
        testing_pred = model.predict(testE)

        print("Training Prediction Accuracy:{}".format(accuracy_score(trainScores, training_pred)))
        print("Training Kappa:{}".format(cohen_kappa_score(trainScores, training_pred)))
        print("Test Prediction Accuracy:{}".format(accuracy_score(testScores, testing_pred)))
        print("Testing Kappa:{}".format(cohen_kappa_score(testScores, testing_pred)))
        accuracies.append(accuracy_score(testing_pred, testScores))
        kappas.append(cohen_kappa_score(testScores, testing_pred))

        with open(out_dir + '/params', 'a') as f:
            f.write(("Training Prediction Accuracy:{}\n".format(accuracy_score(trainScores, training_pred))))
            f.write(("Test Prediction Accuracy:{}\n".format(accuracy_score(testScores, testing_pred))))
            f.write(("Test Kappa:{}\n".format(cohen_kappa_score(testScores, testing_pred))))

    print("Test Average Acc: {}".format(np.mean(accuracies)))
    print("Test Average Kap: {}".format(np.mean(kappas)))
    with open(out_dir + '/params', 'a') as f:
        f.write("test avgs: {}".format(np.mean(accuracies)))
        f.write("test avg kaps: {}".format(np.mean(kappas)))

