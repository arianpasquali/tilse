#!/usr/bin/env python

import argparse
import codecs
import os
import shlex
import sys


try:
    import urllib.request as urlrequest
except ImportError:
    import urllib2.urlopen as urlrequest

import zipfile
import tempfile
import tarfile
import shutil

import pickle
import spacy
import subprocess
from dateutil import parser as date_parser

import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tilse.data import corpora
from tilse.util.temporal_expression import normalize_temporal_expressions
from tilse.util import html_cleaner 
from tilse.util.sentence_segmentation import sentence_segmenter

parser = argparse.ArgumentParser(description='Get and preprocess timeline data.')
parser.add_argument('corpus_name', help='Name of the corpus to download (either timeline17 or crisis).')

args = parser.parse_args()
corpus = args.corpus_name

dataset_lang = corpus.split("_")[-1]
print("dataset lang", dataset_lang)
temp_path = f"/home/dock/workspace/tilse/dataset_covid19/covid19_{dataset_lang}/raw/"
# transform directory structure

path = os.getcwd() + "/" + corpus + "/"

path_raw = os.getcwd() + "/" + corpus + "/raw/"


path_dumped = path + "dumped_corpora/"

os.mkdir(path_dumped)

nlp = spacy.load("en_core_web_sm")
if(dataset_lang == "pt"):
    nlp = spacy.load("pt_core_news_sm")

sources = {
    "en":["guardian","cnn"],
    "pt":["publico","observador"],
}

topics = {}
for topic in os.listdir(path_raw):
    topic_name = topic.replace(f"_{sources[dataset_lang]}.corpus.obj")
    if(topic_name not in topics.keys()):
        topics[topic_name] = []

    topics[topic_name].append(corpora.Corpus.from_folder(path_raw + topic + "/articles/", nlp))

import itertools
# list2d = [[1,2,3], [4,5,6], [7], [8,9]]

for topic in topics.keys():    
    all_docs = []
    for _corpus in topics[topic]:
        all_docs.append(_corpus.docs)

    merged_docs = list(itertools.chain(*all_docs))
    topic_corpora = corpora.Corpus(docs=merged_docs, name=topic)

    
    all_sents = [sent for doc in merged_docs for sent in doc]
    print(f"number of docs : {len(merged_docs)}")
    print(f"number of sentences : {len(all_sents)}")

    # if(len(all_sents) > 1000):
    print(topic, len(all_sents))
    with open(path_dumped + topic + ".corpus.obj", "wb") as my_file:
        pickle.dump(topic_corpora, my_file)
