import argparse
import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import stanza
from nltk.tree import Tree
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import pandas as pd


def cal_lexical(sentences):
    """
    input: a list of sentences that are split in advance using sentence tokenizer
    output: a 6-dimensional vector that represents six features
        (1) avg. # of nouns in a sentence
        (2) avg. # of verbs in a sentence
        (3) avg. # of adjectives in a sentence
        (4) avg. # of unique words in a sentence
        (5) avg. subjectivity score
        (6) avg. # of words with concreteness scores above 3 in a sentence
    """

    # initialization
    total_nouns, total_verbs, total_adjs, total_subjectivity, total_unique_words, total_concreteness = 0, 0, 0, 0, 0, 0
    
    # loop over sentences
    for s in tqdm(sentences, desc='Calculating lexical features', total=len(sentences)):
        # words
        words = word_tokenize(s)
        pos_tags = nltk.pos_tag(words)
        unique_words = set(words)

        noun_count = len([word for word, tag in pos_tags if tag.startswith('NN')])
        verb_count = len([word for word, tag in pos_tags if tag.startswith('VB')])
        adjective_count = len([word for word, tag in pos_tags if tag.startswith('JJ')])

        total_nouns += noun_count
        total_verbs += verb_count
        total_adjs += adjective_count
        total_unique_words += len(unique_words)

        # subjectivity
        blob = TextBlob(s)
        total_subjectivity += blob.sentiment.subjectivity

        # concreteness
        concreteness_df = pd.read_csv('../../data/concreteness.csv')
        concreteness_dict = pd.Series(concreteness_df.Score.values, index=concreteness_df.Word).to_dict()
        total_concreteness += len([word for word in words if concreteness_dict.get(word, 0) > 3])

    # average
    num_sentences = len(sentences)
    avg_nouns = total_nouns / num_sentences
    avg_verbs = total_verbs / num_sentences
    avg_adjs = total_adjs / num_sentences
    avg_subjectivity = total_subjectivity / num_sentences
    avg_unique_words = total_unique_words / num_sentences
    avg_concreteness = total_concreteness / num_sentences

    return [avg_nouns, avg_verbs, avg_adjs, avg_unique_words, avg_subjectivity, avg_concreteness]

def get_all_nodes(tree):
    nodes = []
    for node in tree:
        if isinstance(node, Tree):
            nodes.append(node.label())
            nodes.extend(get_all_nodes(node))
    return nodes

def cal_syntactic(sentences):
    """
    input: a list of sentences that are split in advance using sentence tokenizer
    output: a 5-dimensional vector that represents the probability distribution over the 5 categories
    """

    # initialization
    cat_dict = {
        "SIMPLE": 0,
        "COMPOUND": 0,
        "COMPLEX": 0,
        "COMPLEX-COMPOUND": 0,
        "OTHER": 0
    }
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False)

    # loop over sentences
    for s in tqdm(sentences, desc='Calculating syntactic features', total=len(sentences)):
        doc = nlp(s)
        parse_tree = Tree.fromstring(str(doc.sentences[0].constituency))
        sub_tree = parse_tree[0]
        l_top = [child.label() for child in sub_tree if isinstance(child, Tree)]
        all_nodes = get_all_nodes(parse_tree)

        # get result
        if 'S' in l_top:
            if "SBAR" not in all_nodes:
                cat_dict['COMPOUND'] += 1
            else:
                cat_dict['COMPLEX-COMPOUND'] += 1
        elif 'VP' in l_top:
            if "SBAR" not in all_nodes:
                cat_dict['SIMPLE'] += 1
            else:
                cat_dict['COMPLEX'] += 1
        else:
            cat_dict['OTHER'] += 1
    
    # normalization and return a 5-dimensional vector
    total = sum(cat_dict.values())
    return [cat_dict[key] / total for key in cat_dict]

def cal_surface(sentences):
    """
    input: a list of sentences that are split in advance using sentence tokenizer
    output: a 5-dimensional vector that represents five features
        (1) avg. # of commas in a sentence
        (2) avg. # of semicolons in a sentence
        (3) avg. # of colons in a sentence
        (4) avg. # of words in a sentence
        (5) avg. length of words
    """
    
    # initialization
    commas, semicolons, colons, total_words, total_word_length = 0, 0, 0, 0, 0

    # loop over sentences
    for s in tqdm(sentences, desc='Calculating surface features', total=len(sentences)):
        # punctuation
        commas += s.count(',')
        semicolons += s.count(';')
        colons += s.count(':')

        # words
        words = s.split()
        total_words += len(words)
        total_word_length += sum(len(word) for word in words)

    # average
    num_sentences = len(sentences)
    avg_commas = commas / num_sentences if num_sentences else 0
    avg_semicolons = semicolons / num_sentences if num_sentences else 0
    avg_colons = colons / num_sentences if num_sentences else 0
    avg_words_per_sentence = total_words / num_sentences if num_sentences else 0
    avg_word_length = total_word_length / total_words if total_words else 0

    return [avg_commas, avg_semicolons, avg_colons, avg_words_per_sentence, avg_word_length]

def mean_squared_error(l1, l2):
    vec1 = np.array(l1)
    vec2 = np.array(l2)
    return np.mean((vec1 - vec2) ** 2)

def jensen_shannon_divergence(l1, l2):
    vec1 = np.array(l1)
    vec2 = np.array(l2)
    return jensenshannon(vec1, vec2)

def cal_linguistic_alignment(features1, features2):
    # get three features
    lexical1, syntactic1, surface1 = features1
    lexical2, syntactic2, surface2 = features2

    # use MSE to measure lexical and surface alignment
    lexical_mse = mean_squared_error(lexical1, lexical2)
    surface_mse = mean_squared_error(surface1, surface2)

    # use Jensen-Shannon divergence to measure syntactic alignment
    syntactic_jsd = jensen_shannon_divergence(syntactic1, syntactic2)

    return [lexical_mse, syntactic_jsd, surface_mse]


