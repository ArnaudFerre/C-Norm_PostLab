#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: If you have trained the module_train on a training set (terms associated with concept(s)),
    you can do here a prediction of normalization with a test set (new terms without pre-association with concept).
    If you want to cite this work in your publication or to have more details:
    Ongoing...

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


#######################################################################################################
# Import modules & set up logging
#######################################################################################################

from io import open
from sys import stderr
from optparse import OptionParser
import json
import gzip

import numpy
from scipy.spatial.distance import cosine
import gensim
from tensorflow.keras import layers, models

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from train import normalizeEmbedding
from word2term import getSizeOfVST, getFormOfTerm
from onto import loadOnto, ontoToVec


#######################################################################################################
# Utils
#######################################################################################################

def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename, encoding='utf-8')
    result = json.load(f)
    f.close()
    return result;



def metric_internal(metric):
    if metric == 'cosine':
        return 'euclidean'
    if metric == 'cosine-brute':
        return 'cosine'
    return metric


def metric_norm(metric, concept_vectors):
    if metric == 'cosine':
        return normalize(concept_vectors)
    return concept_vectors


def metric_sim(metric, d, vecTerm, vecConcept):
    if metric == 'cosine':
        return 1 - cosine(vecTerm, vecConcept)
    if metric == 'cosine-brute':
        return 1 - d
    return 1 / d



class VSONN(NearestNeighbors):
    def __init__(self, vso, metric):
        NearestNeighbors.__init__(self, algorithm='auto', metric=metric_internal(metric))
        self.original_metric = metric
        self.vso = vso
        self.concepts = tuple(vso.keys())
        self.concept_vectors = list(vso.values())
        self.fit(metric_norm(metric, self.concept_vectors))

    def nearest_concept(self, vecTerm):
        r = self.kneighbors([vecTerm], 1, return_distance=True)
        #stderr.write('r = %s\n' % str(r))
        d = r[0][0][0]
        idx = r[1][0][0]
        return self.concepts[idx], metric_sim(self.original_metric, d, vecTerm, self.concept_vectors[idx])



#######################################################################################################
# C-Norm predictor
#######################################################################################################

def CNorm_Predictor(vst_onlyTokens, dl_terms, vso, transformationParam, metric, phraseMaxSize, symbol='___'):

    lt_predictions = list()
    result = dict()
    vsoTerms = dict()

    sizeVST = getSizeOfVST(vst_onlyTokens)

    vsoNN = VSONN(vso, metric)
    for id_term in dl_terms.keys():
        x_CNN = numpy.zeros((1, phraseMaxSize, sizeVST))
        x_MLP = numpy.zeros((1, sizeVST))
        for i, token in enumerate(dl_terms[id_term]):
            try:
                x_CNN[0][i] = vst_onlyTokens[token]
                x_MLP[0] += vst_onlyTokens[token]
            except:
                pass
        if len(dl_terms[id_term]) == 0:
            pass
        else:
            x_MLP[0] = x_MLP[0] / len(dl_terms[id_term])

        termForm = getFormOfTerm(dl_terms[id_term], symbol)
        vsoTerms[termForm] = transformationParam.predict([x_MLP, x_CNN])[0][0]

        result[termForm] = vsoNN.nearest_concept(vsoTerms[termForm])

    for id_term in dl_terms.keys():
        termForm = getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions



#######################################################################################################
# Run class:
#######################################################################################################

class Predictor(OptionParser):

    def __init__(self):

        OptionParser.__init__(self, usage='usage: %prog [options]')

        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors', help='path to word vectors file as produced by word2vec')
        self.add_option('--word-vectors-bin', action='store', type='string', dest='word_vectors_bin', help='path to word vectors binary file as produced by word2vec')
        self.add_option('--ontology', action='store', type='string', dest='ontology', help='path to ontology file in OBO format')
        self.add_option('--terms', action='store', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--inputModel', action='store', type='string', dest='model', help='path to the input model from a training')

        self.add_option('--output', action='store', type='string', dest='output', help='file where to write predictions')

        self.add_option('--metric', action='store', type='string', dest='metric', default='cosine', help='distance metric to use (default: %default)')
        self.add_option('--factor', action='store', type='float', dest='factors', default=0.65, help='parent concept weight factor (default=0.6)')
        self.add_option('--normalizedInputs', action='store', type='string', dest='normalizedInputs', default="True", help='unit normalize embeddings if "True" (default: True).')
        self.add_option('--phraseMaxSize', action='store', type='int', dest='phrase_max_size', default=15, help='max considered size of phrases in inputs (default=15).')



    def run(self):

        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.word_vectors is None and options.word_vectors_bin is None:
            raise Exception('missing either --word-vectors or --word-vectors-bin')
        if options.word_vectors is not None and options.word_vectors_bin is not None:
            raise Exception('incompatible --word-vectors or --word-vectors-bin')        
        if options.ontology is None:
            raise Exception('missing --ontology')
        if not(options.terms):
            raise Exception('missing --terms')
        if not(options.model):
            raise Exception('missing --inputModel')
        if not(options.output):
            raise Exception('missing --output')


        # Selected hyperparameters (can have an important influence...):
        print("\nRuning C-Norm with next options (recommended to keep the same as those in the training):")
        print("factor=", options.factors)
        print("normalizedInputs=", options.normalizedInputs)


        # Loading word embeddings:
        if options.word_vectors is not None:
            print("\nloading word embeddings:", options.word_vectors)
            word_vectors = loadJSON(options.word_vectors)
        elif options.word_vectors_bin is not None:
            print("\nloading word embeddings:", options.word_vectors_bin)
            EmbModel = gensim.models.Word2Vec.load(options.word_vectors_bin)
            word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in EmbModel.wv[k])) for k in EmbModel.wv.vocab.keys())

        # Scaling of all embeddings:
        if options.normalizedInputs is not None:
            if options.normalizedInputs == "True":
                print("\nScaling: Unit normalization of input embeddings (recommended)...")
                word_vectors = normalizeEmbedding(word_vectors)
                print("Scaling done.\n")
        else:
            print("No scaling of input embeddings (not recommended, see normalizedInputs option).")

        # Loading ontology:
        print("loading ontology:", options.ontology)
        ontology = loadOnto(options.ontology)

        # Building ontological space
        print("\nBuilding ontological space (recommended to use the same factor that for training)...")
        vso = ontoToVec(ontology, options.factors)
        print("Ontological space built.\n")

        stderr.write('loading terms: %s\n' % options.terms)
        stderr.flush()
        terms = loadJSON(options.terms)

        print("\nloading Tensorflow model:", options.model)
        trained_model = models.load_model(options.model)
        print("Loaded.\n")



        print("C-Norm predicting...")
        prediction = CNorm_Predictor(word_vectors, terms, vso, trained_model, options.metric, options.phrase_max_size)
        print("Prediction done.")



        print("\nwriting predictions:", options.output)
        with open(options.output, 'w') as file:
            dl_prediction = dict()
            file.write('mentionId\tconceptId\tsimilarityScore\n')
            for _, term_id, concept_id, similarity in prediction:
                file.write('%s\t%s\t%f\n' % (term_id, concept_id, similarity))
                dl_prediction[term_id] = [concept_id]
            file.close()



#######################################################################################################
# Test section
#######################################################################################################

if __name__ == '__main__':

    Predictor().run()

