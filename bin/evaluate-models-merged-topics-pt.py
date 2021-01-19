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


# assembleia_da_republica-ar_publico.corpus.obj #remover por causa do ar
# assembleia_da_republica-ar_observador.corpus.obj #remover por causa do ar
# cuidados_intensivos_observador.corpus.obj remover?
# cuidados_intensivos_publico.corpus.obj remover?
# insa ??? nope
# mortes-mortos-vitimas_mortais_publico.corpus.obj #remover 
# mortes-mortos-vitimas_mortais_observador.corpus.obj #remover

consider_corpora = [
"acores_publico",
"acores_observador",
"africa_austral_publico",
"africa_austral_observador",
"africa_do_sul_observador",
"africa_do_sul_publico",
"africa_publico",
"africa_observador",
"alemanha_publico",
"alemanha_observador",
"alentejo_publico",
"alentejo_observador",
"algarve_publico",
"algarve_observador",
"amadora_publico",
"amadora_observador",
"america_latina_publico",
"america_latina_observador",
"ana_mendes_godinho-ana_mendes_observador",
"ana_mendes_godinho-ana_mendes_publico",
"andrew_cuomo-governador_andrew_cuomo-cuomo_publico",
"andrew_cuomo-governador_andrew_cuomo-cuomo_observador",
"angela_merkel_publico",
"angela_merkel_observador",
"antonio_costa-primeiro_ministro_antonio_costa_observador",
"antonio_costa-primeiro_ministro_antonio_costa_publico",
"antonio_lacerda_sales-lacerda_sales-antonio_lacerda-antonio_sales_publico",
"antonio_lacerda_sales-lacerda_sales-antonio_lacerda-antonio_sales_observador",
"argelia_observador",
"argelia_publico",
"argentina_observador",
"argentina_publico",
"australia_observador",
"australia_publico",
"austria_publico",
"austria_observador",
"aveiro_publico",
"aveiro_observador",
"bloco_de_esquerda-be_publico",
"bloco_de_esquerda-be_observador",
"beja_observador",
"beja_publico",
"belgica_observador",
"belgica_publico",
"jair_bolsonaro-bolsonaro_observador",
"jair_bolsonaro-bolsonaro_publico",
"boris_johnson-boris_observador",
"boris_johnson-boris_publico",
"braga_publico",
"braga_observador",
"braganca_observador",
"braganca_publico",
"brasil_observador",
"brasil_publico",
"bruxelas_observador",
"bruxelas_publico",
"california_observador",
"california_publico",
"canada_observador",
"canada_publico",
"casa_branca_publico",
"casa_branca_observador",
"cascais_observador",
"cascais_publico",
"cds-cds_pp_observador",
"cds-cds_pp_publico",
"chega_observador",
"chega_publico",
"china-china_continental_observador",
"china-china_continental_publico",
"coimbra_publico",
"coimbra_observador",
"comissao_de_saude_da_china-comissao_nacional_de_saude_da_china_publico",
"comissao_de_saude_da_china-comissao_nacional_de_saude_da_china_observador",
"comissao_europeia-ce_publico",
"comissao_europeia-ce_observador",
"conselho_de_ministros_publico",
"conselho_de_ministros_observador",
"conselho_europeu_publico",
"conselho_europeu_observador",
"coreia_do_sul_publico",
"coreia_do_sul_observador",
"diario_da_republica_observador",
"diario_da_republica_publico",
"dinamarca_publico",
"dinamarca_observador",
"economia_publico",
"economia_observador",
"eduardo_cabrita-cabrita_publico",
"eduardo_cabrita-cabrita_observador",
"emmanuel_macron-macron_observador",
"emmanuel_macron-macron_publico",
"escocia_observador",
"escocia_publico",
"espanha_observador",
"espanha_publico",
"estado_de_emergencia_publico",
"estado_de_emergencia_observador",
"eua-estados_unidos_da_america-estados_unidos_observador",
"eua-estados_unidos_da_america-estados_unidos_publico",
"europa-europeu_observador",
"europa-europeu_publico",
"evora_observador",
"evora_publico",
"florida_observador",
"florida_publico",
"forcas_armadas_observador",
"forcas_armadas_publico",
"franca_observador",
"franca_publico",
"guarda_nacional_republicana-gnr_observador",
"guarda_nacional_republicana-gnr_publico",
"graca_freitas_publico",
"graca_freitas_observador",
"grecia-a_grecia_publico",
"grecia-a_grecia_observador",
"holanda_observador",
"holanda_publico",
"hong_kong_observador",
"hong_kong_publico",
"hungria_observador",
"hungria_publico",
"ilha_de_sao_miguel_publico",
"ilha_de_sao_miguel_observador",
"india_observador",
"india_publico",
"infarmed_publico",
"infarmed_observador",
"iniciativa_liberal-il_publico",
"iniciativa_liberal-il_observador",
"instituto_de_administracao_da_saude_observador",
"instituto_de_administracao_da_saude_publico",
"instituto_robert_koch-robert_koch-rki-instituto_robert_observador",
"instituto_robert_koch-robert_koch-rki-instituto_robert_publico",
"irao_publico",
"irao_observador",
"irlanda_publico",
"irlanda_observador",
"israel_publico",
"israel_observador",
"italia_publico",
"italia_observador",
"japao_publico",
"japao_observador",
"joe_biden-biden_publico",
"joe_biden-biden_observador",
"lisboa-cidade_de_lisboa_observador",
"lisboa-cidade_de_lisboa_publico",
"londres_publico",
"londres_observador",
"loures_observador",
"loures_publico",
"macau_publico",
"macau_observador",
"madeira-arquipelago_da_madeira-regiao_autonoma_da_madeira_publico",
"madeira-arquipelago_da_madeira-regiao_autonoma_da_madeira_observador",
"madrid-comunidade_de_madrid_observador",
"madrid-comunidade_de_madrid_publico",
"marcelo_rebelo_de_sousa-rebelo_de_sousa_observador",
"marcelo_rebelo_de_sousa-rebelo_de_sousa_publico",
"marta_temido_publico",
"marta_temido_observador",
"matosinhos_observador",
"matosinhos_publico",
"mexico_observador",
"mexico_publico",
"ministerio_da_administracao_interna-mai_observador",
"ministerio_da_administracao_interna-mai_publico",
"ministerio_da_educacao_observador",
"ministerio_da_educacao_publico",
"ministerio_da_saude_observador",
"ministerio_da_saude_do_brasil_publico",
"moscovo_observador",
"moscovo_publico",
"natal_publico",
"natal_observador",
"nova_iorque-cidade_de_nova_iorque_observador",
"nova_iorque-cidade_de_nova_iorque_publico",
"odivelas_observador",
"odivelas_publico",
"oms-organizacao_mundial_de_saude-organizacao_mundial_da_saude_observador",
"oms-organizacao_mundial_de_saude-organizacao_mundial_da_saude_publico",
"onu-organizacao_das_nacoes_unidas-nacoes_unidas_observador",
"onu-organizacao_das_nacoes_unidas-nacoes_unidas_publico",
"ordem_dos_medicos_publico",
"ordem_dos_medicos_observador",
"ordem_dos_enfermeiros_publico",
"ordem_dos_enfermeiros_observador",
"ovar_observador",
"ovar_publico",
"pais_de_gales_observador",
"pais_de_gales_publico",
"paises_baixos_publico",
"paises_baixos_observador",
"pan_publico",
"pan_observador",
"pandemia_publico",
"pandemia_observador",
"paris_publico",
"paris_observador",
"pascoa_publico",
"pascoa_observador",
"pcp_observador",
"pcp_publico",
"pedro_sanchez_observador",
"pedro_sanchez_publico",
"pequim_observador",
"pequim_publico",
"peru_publico",
"peru_observador",
"polonia_observador",
"polonia_publico",
"porto_publico",
"porto_observador",
"portugal-portugueses_observador",
"portugal-portugueses_publico",
"prevencao_de_doencas_da_uniao_africana_observador",
"prevencao_de_doencas_da_uniao_africana_publico",
"profissionais_de_saude_observador",
"profissionais_de_saude_publico",
"partido_socialista-ps_observador",
"partido_socialista-ps_publico",
"psd_observador",
"psd_publico",
"policia_de_seguranca_publica-psp_observador",
"policia_de_seguranca_publica-psp_publico",
"reguengos_de_monsaraz-reguengos_observador",
"reguengos_de_monsaraz-reguengos_publico",
"reino_unido_observador",
"reino_unido_publico",
"republica_checa_publico",
"republica_checa_observador",
"rio_de_janeiro_observador",
"rio_de_janeiro_publico",
"russia_publico",
"russia_observador",
"santarem_observador",
"santarem_publico",
"sao_miguel_observador",
"sao_miguel_publico",
"sao_paulo_publico",
"sao_paulo_observador",
"saude_publica_observador",
"saude_publica_publico",
"seguranca_social_publico",
"seguranca_social_observador",
"servico_nacional_de_saude-sns_observador",
"servico_nacional_de_saude-sns_publico",
"setubal_publico",
"setubal_observador",
"sintra_publico",
"sintra_observador",
"stayaway_covid-aplicacao_stayaway_covid-app_stayaway_covid_publico",
"stayaway_covid-aplicacao_stayaway_covid-app_stayaway_covid_observador",
"suecia_publico",
"suecia_observador",
"suica_publico",
"suica_observador",
"tap_publico",
"tap_observador",
"tedros_adhanom_ghebreyesus-ghebreyesus_observador",
"tedros_adhanom_ghebreyesus-ghebreyesus_publico",
"texas_observador",
"texas_publico",
"trabalho_observador",
"trabalho_publico",
"donald_trump-trump_publico",
"donald_trump-trump_observador",
"uniao_europeia-ue-estados_membros_publico",
"uniao_europeia-ue-estados_membros_observador",
"universidade_de_oxford_observador",
"universidade_de_oxford_publico",
"universidade_johns_hopkins-johns_hopkins-universidade_john_hopkins_observador",
"universidade_johns_hopkins-johns_hopkins-universidade_john_hopkins_publico",
"ursula_von_der_leyen-ursula_von_der_publico",
"ursula_von_der_leyen-ursula_von_der_observador",
"vacina-vacinas_observador",
"vacina-vacinas_publico",
"venezuela_observador",
"venezuela_publico",
"vila_real_observador",
"vila_real_publico",
"wuhan_observador",
"wuhan_publico",
"xangai_observador",
"xangai_publico"
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
corpus = "/home/dock/workspace/tilse/covid19_pt"
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


