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

ignore_topics = """us-the_us-united_states-the_united_states-american-americans_guardian
us-the_us-united_states-the_united_states-american-americans_cnn
confirmed_cases_guardian
million_coronavirus_cases_guardian
million_coronavirus_cases_cnn
confirmed_cases_cnn
cases_cnn
rome_guardian
wales_guardian
wales_cnn
rome_cnn
christmas_cnn
christmas_
cases_guardian
death_toll_guardian
death_toll_cnn
britain-british_cnn
britain-british_guardian
eu_guardian
eu_cnn
the_us_food_and_drug_administration_guardian
the_us_food_and_drug_administration_cnn
brazil_death_toll_cnn
brazil_death_toll_guardian"""

ignore_topics.split("\n")

topics = {}
# sort by name
# save each double

all_topics = os.listdir(path_raw)
all_topics = sorted(all_topics)

source_a = []
source_b = []

if(dataset_lang == "en"):
    source_a = [corpus_name for corpus_name in all_topics if "guardian" in corpus_name]
    source_b = [corpus_name for corpus_name in all_topics if "cnn" in corpus_name]
elif(dataset_lang == "pt"):
    source_a = [corpus_name for corpus_name in all_topics if "publico" in corpus_name]
    source_b = [corpus_name for corpus_name in all_topics if "observador" in corpus_name]

import itertools
for idx, topic_a in enumerate(source_a):
    # ignorar topicos da lista
    if topic_a in ignore_topics:
        print("ignorando topico", topic_a)
        continue

    topic_name = topic_a.replace(f"_{sources[dataset_lang]}.corpus.obj","")
    # if(topic_name not in topics.keys()):
    #     topics[topic_name] = []
    print("topic_name", topic_name)

    topic_b = source_b[idx]
    corpus_a = corpora.Corpus.from_folder(path_raw + topic_a + "/articles/", nlp)
    corpus_b = corpora.Corpus.from_folder(path_raw + topic_b + "/articles/", nlp)

    all_docs = corpus_a.docs + corpus_b.docs
    merged_docs = list(itertools.chain(*all_docs))
    merged_corpus = corpora.Corpus(docs=merged_docs, name=topic_name)
    
    all_sents = [sent for doc in merged_docs for sent in doc]
    print(f"topic: {topic_name} | n_docs: {len(merged_docs)} | n_sentences :{len(all_sents)}")
    with open(path_dumped + topic_name + ".corpus.obj", "wb") as my_file:
        pickle.dump(merged_corpus, my_file)

# list2d = [[1,2,3], [4,5,6], [7], [8,9]]

# for topic in topics.keys():    
#     all_docs = []
#     for _corpus in topics[topic]:
#         all_docs.append(_corpus.docs)

#     merged_docs = list(itertools.chain(*all_docs))
#     topic_corpora = corpora.Corpus(docs=merged_docs, name=topic)

    
#     all_sents = [sent for doc in merged_docs for sent in doc]
#     print(f"number of docs : {len(merged_docs)}")
#     print(f"number of sentences : {len(all_sents)}")

#     # if(len(all_sents) > 1000):
#     print(topic, len(all_sents))
#     with open(path_dumped + topic + ".corpus.obj", "wb") as my_file:
#         pickle.dump(topic_corpora, my_file)
