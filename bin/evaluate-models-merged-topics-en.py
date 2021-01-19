#!/usr/bin/env python
import csv
import argparse
import codecs
import json
import logging
import os
from os import listdir
import pickle
import pprint
import sys
from collections import defaultdict

import spacy

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tilse.models.regression import regression
from tilse.models.submodular import submodular, upper_bounds
from tilse.models.chieu import chieu
from tilse.models.random import random

from tilse.data import timelines
from tilse.evaluation import rouge
from tilse.data.corpora import Corpus
from tilse.data.documents import Document

# from install_external_nlp_models import load_spacy_model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

parser = argparse.ArgumentParser(description='Predict timelines using public `timeline17 dataset.')
parser.add_argument('--config_file', help='config JSON file containing parameters.')
parser.add_argument('--consider_headlines_only', help='Consider only headlines', action='store_true')

args = parser.parse_args()

# ignore_topics = """us-the_us-united_states-the_united_states-american-americans_guardian
# us-the_us-united_states-the_united_states-american-americans_cnn
# confirmed_cases_guardian
# million_coronavirus_cases_guardian
# million_coronavirus_cases_cnn
# confirmed_cases_cnn
# cases_cnn
# rome_guardian
# wales_guardian
# wales_cnn
# rome_cnn
# christmas_cnn
# christmas_
# cases_guardian
# death_toll_guardian
# death_toll_cnn
# britain-british_cnn
# britain-british_guardian
# eu_guardian
# eu_cnn
# the_us_food_and_drug_administration_guardian
# the_us_food_and_drug_administration_cnn
# brazil_death_toll_cnn
# brazil_death_toll_guardian"""

# ignore_topics.split("\n")

# consider_corpora = {
# # "uk":["uk","england","britain-british"],
# "california":["california"],
# "white_house":["white_house-the_white_house"],
# "china":["china-mainland_china-chinese"],
# "mexico":["mexico"],
# "asia":["asia"],
# "cdc":["cdc-centers_for_disease_control_and_prevention"],
# "italy":["italy-italian"],
# "wuhan":["wuhan"],
# "australia":["australia-australian"],
# "florida":["florida"],
# "germany":["germany"],
# "joe biden":["joe_biden"],
# "russia":["russia"],
# "un":["un"],
# "victoria":["victoria"],
# "pfizer":["pfizer"],
# "france":["france-french"],
# "canada":["canada"],
# "india":["india"],
# "japan":["japan"],
# "trump":["donald_trump-trump-president_donald_trump"],
# "europe":["europe-european"],
# "spain":["spain"],
# "latin america":["latin_america"],
# "london":["london"],
# "coronavirus":["coronavirus"],
# "south korea":["south_korea"],
# "hong kong":["hong_kong"],
# "beijing":["beijing"],
# "texas":["texas"],
# "johns hopkins university":["johns_hopkins-johns_hopkins_university"]

# "brazil":["brazil-brazilian"],
# "boris johnson":["boris_johnson-prime_minister_boris"],
# "new york":["new_york-new_york_city"],
# "world health organization":["who-world_health_organization-the_world_health_organization"],
# }


consider_corpora = [
"california_cnn",
"white_house-the_white_house_cnn",
"china-mainland_china-chinese_cnn",
"mexico_cnn",
"asia_cnn",
"cdc-centers_for_disease_control_and_prevention_cnn",
"italy-italian_cnn",
"wuhan_cnn",
"australia-australian_cnn",
"florida_cnn",
"germany_cnn",
"joe_biden_cnn",
"russia_cnn",
"un_cnn",
"victoria_cnn",
"pfizer_cnn",
"france-french_cnn",
"canada_cnn",
"india_cnn",
"japan_cnn",
"donald_trump-trump-president_donald_trump_cnn",
"europe-european_cnn",
"spain_cnn",
"latin_america_cnn",
"london_cnn",
"coronavirus_cnn",
"south_korea_cnn",
"hong_kong_cnn",
"beijing_cnn",
"texas_cnn",
"johns_hopkins-johns_hopkins_university_cnn",
"brazil-brazilian_cnn",
"boris_johnson-prime_minister_boris_cnn",
"new_york-new_york_city_cnn",
"who-world_health_organization-the_world_health_organization_cnn",

"california_guardian",
"white_house-the_white_house_guardian",
"china-mainland_china-chinese_guardian",
"mexico_guardian",
"asia_guardian",
"cdc-centers_for_disease_control_and_prevention_guardian",
"italy-italian_guardian",
"wuhan_guardian",
"australia-australian_guardian",
"florida_guardian",
"germany_guardian",
"joe_biden_guardian",
"russia_guardian",
"un_guardian",
"victoria_guardian",
"pfizer_guardian",
"france-french_guardian",
"canada_guardian",
"india_guardian",
"japan_guardian",
"donald_trump-trump-president_donald_trump_guardian",
"europe-european_guardian",
"spain_guardian",
"latin_america_guardian",
"london_guardian",
"coronavirus_guardian",
"south_korea_guardian",
"hong_kong_guardian",
"beijing_guardian",
"texas_guardian",
"johns_hopkins-johns_hopkins_university_guardian",
"brazil-brazilian_guardian",
"boris_johnson-prime_minister_boris_guardian",
"new_york-new_york_city_guardian",
"who-world_health_organization-the_world_health_organization_guardian"

]

modelbase_dir = "configs/tlscovid19/"

if not(args.config_file):
    # read all configs
    configs = listdir(modelbase_dir)
else:
    configs = [args.config_file]

is_headline_based = args.consider_headlines_only

temp_reference_timelines = defaultdict(list)
news_corpora = {}
reference_timelines = {}

# corpus = "timeline17"
# dataset_lang = corpus.split("_")[-1]
# print("dataset lang", dataset_lang)
# logging.info("loading spacy language model")
# nlp = spacy.load("en_core_web_sm")
# if(dataset_lang == "pt"):
#     nlp = spacy.load("pt_core_news_sm")

keyword_mapping =  {
    # "bpoil": ["bp", "oil", "spill"],
    # "egypt": ["egypt", "egyptian"],
    # "finan": ["financial", "economic", "crisis"],
    # "h1n1": ["h1n1", "swine", "flu"],
    # "haiti": ["haiti", "quake", "earthquake"],
    # "iraq": ["iraq", "iraqi"],
    # "libya": ["libya", "libyan"],
    # "mj": ["michael", "jackson"],
    # "syria": ["syria", "syrian"],
    # "yemen": ["yemen"]
  }

# config_file = modelbase_dir + _config
# config = json.load(open(config_file))

# corpus = config["corpus"]
corpus = "/home/dock/workspace/tilse/covid19_en"
raw_directory = corpus + "/raw/"
dumped_corpora_directrory = corpus + "/dumped_corpora_by_source/"

# for topic in sorted(os.listdir(raw_directory)):
for topic in sorted(consider_corpora):

    logging.debug(topic)

    # ignorar topicos da lista
    # if topic in ignore_topics:
    #     print("ignorando topico", topic)
    #     continue

    dataset_path = dumped_corpora_directrory + topic + ".corpus.obj"
    logging.info("loading corpus " + dataset_path)
    logging.info("this may take a while ...")

    news_corpora[topic] = pickle.load(open(dataset_path, "rb"))
    news_corpora[topic].name = topic

    # if(is_headline_based):
    #     docs_headlines = []
    #     for doc in news_corpora[topic]: 
                
    #         try:
    #             docs_headlines.append(Document(doc_id=doc.doc_id, metadata=doc.metadata, publication_date=doc.publication_date, sentences=[doc.sentences[0]]))            
    #         except:
    #             docs_headlines.append(Document( publication_date=doc.publication_date, sentences=[doc.sentences[0]]))

    #     news_corpora[topic] = Corpus(docs=docs_headlines, name=topic)

    sents = [sent for doc in news_corpora[topic] for sent in doc]
    logging.debug("number of docs : {}".format (len(news_corpora[topic].docs)))
    logging.debug("number of sentences : {}".format(len(sents)))

    # filter dataset by keywords
    # if keyword_mapping is not None and keyword_mapping[topic] is not None:
    #     news_corpora[topic] = news_corpora[topic].filter_by_keywords_contained(keyword_mapping[topic])

    # read groundtruth timelines
    for filename in sorted(list(os.listdir(raw_directory + "/" + topic + "/timelines/"))):
        full_path = raw_directory + "/" + topic + "/timelines/" + filename

        temp_reference_timelines[topic].append(
            timelines.Timeline.from_file(codecs.open(full_path, "r", "utf-8", "replace"))
        )

for topic in temp_reference_timelines:
    reference_timelines[topic] = timelines.GroundTruth(temp_reference_timelines[topic])

evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"], beta=1)

model_avg_scores = {}

for _config in configs:
    # config_file = modelbase_dir + _config
    config_file = _config
    print(_config)
    print(config_file)
    config = json.load(open(config_file))
    config["rouge_computation"]=  "reimpl"
    
    logging.info(config)

    algorithm = None

    if config["algorithm"] == "chieu":
        algorithm = chieu.Chieu(config, evaluator)
    elif config["algorithm"] == "regression":
        algorithm = regression.Regression(config, evaluator)
    elif config["algorithm"] == "random":
        algorithm = random.Random(config, evaluator)
    elif config["algorithm"] == "submodular":
        algorithm = submodular.Submodular(config, evaluator)
    elif config["algorithm"] == "upper_bound":
        algorithm = upper_bounds.UpperBounds(config, evaluator)

    returned_timelines, scores = algorithm.run(news_corpora, reference_timelines)

    groundtruths = {}

    pprint.pprint(config)
    scores_to_print = "\t".join(("\n" + str(scores)).splitlines(True))
    logging.debug(scores_to_print)

    model_name = config["name"]

    model_avg_scores.update({
        model_name:{
            "scores":{
                "date_select":scores.date_mapping["average_score"]["f_score"],
                "concat_r1":scores.mapping["average_score"]["concat"]["rouge_1"]["f_score"],
                "concat_r2":scores.mapping["average_score"]["concat"]["rouge_2"]["f_score"],

                "agreement_r1":scores.mapping["average_score"]["agreement"]["rouge_1"]["f_score"],
                "agreement_r2":scores.mapping["average_score"]["agreement"]["rouge_2"]["f_score"],

                "align_date_m_one_r1":scores.mapping["average_score"]["align_date_content_costs_many_to_one"]["rouge_1"]["f_score"],
                "align_date_m_one_r2":scores.mapping["average_score"]["align_date_content_costs_many_to_one"]["rouge_2"]["f_score"],
            }
        }
    })
    output_filename = ""
    output_filename += config["name"] 

    if(is_headline_based):
        output_filename = config["name"] +'_headlines'

    output_filename = output_filename +'_scores.csv'

    with open(output_filename, mode='w') as scores_file:
        # setup csv writer
        scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # header
        scores_writer.writerow(["model","mode","metric","precision","recall","f_score"])

        topics = sorted(list(scores.mapping.keys()))
        modes = [k for k in scores.mapping[list(topics)[0]]]

        # header
        scores_writer.writerow([model_name,
                                "date selection",
                                None,
                                "%.3f" % scores.date_mapping["average_score"]["precision"],
                                "%.3f" % scores.date_mapping["average_score"]["recall"],
                                "%.3f" % scores.date_mapping["average_score"]["f_score"]])
        for mode in modes:
            for metric in sorted(scores.mapping[list(topics)[0]][mode]):

                scores_writer.writerow([model_name,
                                        mode,
                                        metric,
                                        "%.3f" % scores.mapping["average_score"][mode][metric]["precision"],
                                        "%.3f" % scores.mapping["average_score"][mode][metric]["recall"],
                                        "%.3f" % scores.mapping["average_score"][mode][metric]["f_score"]])

logging.info("### Final scores #####################")
logging.info(model_avg_scores)

outputfile_name = "_all_models_evaluation"

if(is_headline_based):
    output_filename += '_headlines'

output_filename += '_scores.csv'

with open(outputfile_name, mode='w') as scores_file:
    # setup csv writer
    scores_writer = csv.writer(scores_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # header
    scores_writer.writerow(["headline_based", 
                                    "model", 
                                    "concat_r1",
                                    "concat_r2",
                                    "agreement_r1",
                                    "agreement_r2",
                                    "align_date_m_one_r1",
                                    "align_date_m_one_r2",
                                    "date_selection"])

    # write average scores for every model 
    for model_ in model_avg_scores.keys():

        scores_writer.writerow([str(is_headline_based),
                                model_,
                                model_avg_scores[model_]["scores"]["concat_r1"],
                                model_avg_scores[model_]["scores"]["concat_r2"],
                                model_avg_scores[model_]["scores"]["agreement_r1"],
                                model_avg_scores[model_]["scores"]["agreement_r2"],
                                model_avg_scores[model_]["scores"]["align_date_m_one_r1"],
                                model_avg_scores[model_]["scores"]["align_date_m_one_r2"],
                                model_avg_scores[model_]["scores"]["date_select"]])

logging.info("Done!")


