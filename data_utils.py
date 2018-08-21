import re
import os as os
import collections
import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

def load_open_response_data(training_path, pid=''):
    training_df = pd.read_csv(training_path,encoding='latin1')
    # get only rows for a specific problem id
    if pid != '':
        training_df = training_df[training_df['problem_id'].isin([pid])]
    essay_list = []
    question_list = []
    essays = training_df['answer_text']
    scores = 4*training_df['correct']
    problem_id = training_df['problem_id']
    question =training_df['problem_text']
    temp_score = scores.tolist()
    count_one = np.sum(temp_score)
    #for i in range(len(temp_score)):
    #    if temp_score[i] >= 0.5:
    #        temp_score[i] = 1
    #        count_one = count_one + 1
    #    else:
    #        temp_score[i] = 0
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        #essay_list.append([w for w in tokenize(essay) if is_ascii(w)])
        essay_list.append(tokenize(essay))

    for idq, question in question.iteritems():
        question = clean_str(question)
        question_list.append(tokenize(question))
    return essay_list, temp_score, problem_id, count_one, question_list

def load_open_response_data_per_assignment(training_path, pid, asid):
    training_df = pd.read_csv(training_path,encoding='latin1')
    # get only rows for a specific problem id
    if pid != '':
        training_df = training_df[training_df['problem_id'].isin([pid])]
        training_df = training_df[training_df['assignment_id'].isin([asid])]
    essay_list = []
    question_list = []
    essays = training_df['answer_text']
    scores = 4*training_df['correct']
    problem_id = training_df['problem_id']
    question =training_df['problem_text']
    temp_score = scores.tolist()
    count_one = np.sum(temp_score)
    #for i in range(len(temp_score)):
    #    if temp_score[i] >= 0.5:
    #        temp_score[i] = 1
    #        count_one = count_one + 1
    #    else:
    #        temp_score[i] = 0
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        #essay_list.append([w for w in tokenize(essay) if is_ascii(w)])
        essay_list.append(tokenize(essay))

    for idq, question in question.iteritems():
        question = clean_str(question)
        question_list.append(tokenize(question))
    return essay_list, temp_score, problem_id, count_one, question_list

def load_detail_data(training_path, pid=''):
    training_df = pd.read_csv(training_path,encoding='latin1')
    # get only rows for a specific problem id
    if pid != '':
        training_df = training_df[training_df['PID'].isin([pid])]
    essay_list = []
    essays = training_df['Text']
    scores = training_df['PredictionCorrect']
    problem_id = training_df['PID']
    predictions = training_df['Predicted'].tolist()
    true_label = training_df['Label'].tolist()
    temp_score = scores.tolist()
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        essay_list.append(tokenize(essay))
    for l in range(len(true_label)):
        true_label[l] = int(true_label[l])
        predictions[l] = int(predictions[l])

    return essay_list, temp_score, problem_id, predictions, true_label

def load_open_response_data_and_comments(training_path, pid=''):
    training_df = pd.read_csv(training_path,encoding='latin1')
    # get only rows for a specific problem id
    if pid != '':
        training_df = training_df[training_df['problem_id'].isin([pid])]
    essay_list = []
    question_list = []
    comment_list = []
    essays = training_df['answer_text']
    scores = 4*training_df['correct']
    comments = training_df['teacher_comment']
    problem_id = training_df['problem_id']
    question =training_df['problem_text']
    temp_score = scores.tolist()
    count_one = np.sum(temp_score)
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        essay_list.append(tokenize(essay))

    for idq, question in question.iteritems():
        question = clean_str(question)
        question_list.append(tokenize(question))

    for idc, comment in comments.iteritems():
        comment = clean_str(comment)
        comment_list.append(tokenize(comment))
    return essay_list, temp_score, problem_id, count_one, question_list, comment_list

def load_open_response_data_character_level(training_path):
    training_df = pd.read_csv(training_path, encoding='latin1')

    essay_list = []
    problem_list = []
    question_list = []
    essays = training_df['answer_text']
    scores = training_df['correct']
    problem_ids = training_df['problem_id']
    temp_score = scores.tolist()

    for idx, essay in essays.iteritems():
        essay_list.append(list(essay))

    for idy, pid in problem_ids.iteritems():
        problem_list.append(list(str(pid)))

    return essay_list, temp_score, problem_ids

def load_pidlist_by_assignment(training_path):
    training_df = pd.read_csv(training_path, encoding='latin1')
    as_pid_tuples = []
    pids = set(training_df['problem_id'])
    for pid in pids:
        target_problems = training_df[training_df['problem_id'].isin([pid])]
        assignment_ids = set(target_problems.assignment_id)
        for asid in assignment_ids:
            as_pid_tuples.append((asid, pid))
    return as_pid_tuples

def load_pidlist(training_path):
    training_df = pd.read_csv(training_path, encoding='latin1')
    return list(set(training_df['problem_id']))


def load_training_data(training_path):
    training_df = pd.read_csv(training_path, delimiter='\t')
    resolved_score_list = []
    essay_list = []
    for essay_set in range(1, 4):
        # resolved score for essay set 1
        resolved_score = training_df[training_df['essay_set'] == essay_set]['domain1_score']
        #convert resoved_score to list
        temp_score = resolved_score.tolist()

        #essay1 max score = 12
        if (essay_set == 1):
            for i in range(len(temp_score)):
                if temp_score[i] > 6:
                    temp_score[i] = 1
                else:
                    temp_score[i] = 0
        #essay2 max score = 6
        elif (essay_set == 2):
            for i in range(len(temp_score)):
                if temp_score[i] > 3:
                    temp_score[i] = 1
                else:
                    temp_score[i] = 0
        #essay3 max score = 3
        elif (essay_set == 3):
            for i in range(len(temp_score)):
                if temp_score[i] > 1:
                    temp_score[i] = 1
                else:
                    temp_score[i] = 0

        for ite in temp_score:
            resolved_score_list.append(ite)

        essays = training_df[training_df['essay_set'] == essay_set]['essay']

        # turn an essay to a list of words
        for idx, essay in essays.iteritems():
            essay = clean_str(essay)
            #essay_list.append([w for w in tokenize(essay) if is_ascii(w)])
            essay_list.append(TreebankWordTokenizer().tokenize(essay))
    return essay_list, resolved_score_list
           #resolved_score.tolist()

def load_open_response_data_char_count(training_path, pid=''):
    essay_list, temp_score, problem_id, count_one, question_list = load_open_response_data(training_path, pid)
    for i in range(len(essay_list)):
        words = ""
        for word in essay_list[i]:
            words += word
        essay_list[i] = len(words)
    return essay_list, temp_score, problem_id, count_one, question_list

    
def load_glove_42b(dim=300):
    word2vec = []
    word_idx = {}
    # first word is nil
    word2vec.append([0]*dim)
    print("==> loading glove")
    count = 1
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove.42B." + str(dim) + "d.txt"), encoding='utf8') as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = np.array([float(val) for val in l[1:]])#list(map(float, l[1:]))
            word_idx[word] = count
            word2vec.append(vector)
            count += 1
            # if count > 255:
            #     break

    print("==> glove is loaded")
    word2vec = np.array(word2vec)
    return word_idx, word2vec


def load_glove_6b(dim=300):
    word2vec = []
    word_idx = {}
    # first word is nil
    word2vec.append([0] * dim)
    print("==> loading glove")
    count = 1
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove.6B." + str(dim) + "d.txt"),
              encoding='utf8') as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = np.array([float(val) for val in l[1:]])  # list(map(float, l[1:]))
            word_idx[word] = count
            word2vec.append(vector)
            count += 1
            # if count > 255:
            #     break

    print("==> glove is loaded")
    word2vec = np.array(word2vec)
    return word_idx, word2vec

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    #>>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    #>>> tokenize('I don't know')
    ['I', 'don', '\'', 'know']
    '''
    raw_tokens = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    fixed_tokens = []
    # re-iterate through the list to break down any numeric entries
    for t in raw_tokens:
        if not t.isalpha():
            # isAlp = t[0].isalpha()
            # i = 0
            # for j in range(1, len(t)):
            #     if (t[j].isalpha() and not isAlp) or (not t[j].isalpha() and isAlp):
            #         fixed_tokens.append(t[i:j])
            #         i=j
            #         isAlp = not isAlp
            #     if t[i:]:
            #         fixed_tokens.append(t[i:])
           fixed_tokens += list(t)
        else:
            fixed_tokens.append(t)
    return fixed_tokens

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = str(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\-\+\=\/\*\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"=", " = ", string)
    string = re.sub(r"/", " / ", string)
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"-", " - ", string)
    string = re.sub(r"\*", " \* ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()

    return string.strip()
   # return string.strip().lower()

# data is DataFrame
def vectorize_data(data, word_idx, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        for w in essay:
            # quick & sloppy conversion of some basic math symbols
            if w == "=":
                w = "equals"
            elif w == "+":
                w = "plus"
            elif w == "-":
                w = "minus"
            elif w == "/":
                w = "over"
            elif w == "*":
                w = "times"

            if w in word_idx:
                wl.append(word_idx[w])
            else:
                # print('{} is not in vocab'.format(w))
                wl.append(0)
        wl += [0]*ls
        E.append(wl)
    E = np.array(E)
    return E

def one_hot_data(data, word_idx, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        wl += [0]*len(word_idx)
        for w in essay:
            # quick & sloppy conversion of some basic math symbols
            if w == "=":
                w = "equals"
            elif w == "+":
                w = "plus"
            elif w == "-":
                w = "minus"
            elif w == "/":
                w = "over"
            elif w == "*":
                w = "times"

            if w in word_idx:
                wl[word_idx[w]] = 1
        E.append(wl)
    E = np.array(E)
    return E
