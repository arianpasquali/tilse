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

for topic in os.listdir(path_raw):
    corpus = corpora.Corpus.from_folder(path_raw + topic + "/articles/", nlp)

    with open(path_dumped + topic + ".corpus.obj", "wb") as my_file:
        pickle.dump(corpus, my_file)
