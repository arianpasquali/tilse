from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse.linalg as sp_linalg
from networkx import DiGraph
import networkx
import os
import sys
import numpy as np
import networkx as nx
import itertools as it
import random
import math

# _tr = {} 
# for tr in tr_scores.keys(): 
#     _tr[tr[0]] = tr_scores[tr]
# 
# for sent in sent_tokens: 
#   print(sent, calculate_informativeness(sent, _tr))
# 
# input uma sentence com uma lista de tokens
# output score da sentenca
import logging
import string
import nltk

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS.add(",")
STOPWORDS.add(".")
STOPWORDS.add("''")
STOPWORDS.add("``")
STOPWORDS.update(string.punctuation)
STOPWORDS.add("'s")
STOPWORDS.add("n't")
STOPWORDS.add("-LRB-")
STOPWORDS.add("-RRB-")
STOPWORDS.add("said")

def calculate_informativeness(sent, tr_scores):
    _tr = {} 
    for tr in tr_scores.keys(): 
        _tr[tr[0]] = tr_scores[tr]

    informativeness_score = 0
    for token in set(sent):
        informativeness_score += _tr.get(token.lower(), 0)

    return informativeness_score

def calculate_keyword_text_rank(sentences, window_size=None):
    graph = nx.Graph()

    print("Computing TextRank for {} sentences".format(len(sentences)))

    for sent in sentences:
        context = sent
        for token in sent:
            tok = token.content.lower()
            pos = token.pos

        # for tok, pos in sent:
            if tok.lower() in STOPWORDS or not tok.isalnum():
                continue
            if not pos.lower()[0] in "vn":
                continue
            graph.add_node((tok, pos))
            # for context_tok, context_pos in context:
            for context_token in context:
                context_tok = context_token.content.lower()
                context_pos = context_token.pos
                if context_tok == tok and context_pos == pos:
                    continue
                if (context_tok.lower() not in STOPWORDS and context_tok.isalnum() and context_pos.lower()[0] in "vn"):
                    graph.add_edge((context_tok, context_pos), (tok, pos))

    pr = nx.pagerank_scipy(graph)

    del graph

    return pr