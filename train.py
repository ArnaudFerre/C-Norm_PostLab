#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Training module for C-Norm method

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

from optparse import OptionParser
import json
import gzip
from os.path import dirname, exists
from os import makedirs

import numpy
import gensim
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend

from word2term import getSizeOfVST
from onto import loadOnto, ontoToVec

#######################################################################################################
# Utils
#######################################################################################################

# Normalization of token embeddings:
def normalizeEmbedding(vst_onlyTokens):
    for token in vst_onlyTokens.keys():
        vst_onlyTokens[token] = vst_onlyTokens[token] / numpy.linalg.norm(vst_onlyTokens[token])
    return vst_onlyTokens


# CHoose with getMatricForCNN...?
def prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize):

    #ToDo: Keep a constant size of the input matrix between train and prediction
    nbTerms = len(dl_terms.keys())
    sizeVST = getSizeOfVST(vst_onlyTokens)
    sizeVSO = getSizeOfVST(vso)

    X_train = numpy.zeros((nbTerms, phraseMaxSize, sizeVST))
    Y_train = numpy.zeros((nbTerms, 1, sizeVSO))

    l_unkownTokens = list()
    l_uncompleteExpressions = list()

    for i, id_term in enumerate(dl_associations.keys()):
        # stderr.write('id_term = %s\n' % str(id_term))
        # stderr.write('len(dl_associations[id_term]) = %d\n' % len(dl_associations[id_term]))

        for id_concept in dl_associations[id_term]:
            Y_train[i][0] = vso[id_concept]
            for j, token in enumerate(dl_terms[id_term]):
                if j < phraseMaxSize:
                    if token in vst_onlyTokens.keys():
                        X_train[i][j] = vst_onlyTokens[token]
                    else:
                        l_unkownTokens.append(token)
                else:
                    l_uncompleteExpressions.append(id_term)
            break # Because it' easier to keep only one concept per mention (mainly to calculate size of matrix).
            # ToDo: switch to object to include directly size with these structures.

    return X_train, Y_train, l_unkownTokens, l_uncompleteExpressions


#
def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        # f = open(filename, encoding='utf-8')
        f = open(filename, "r", encoding="utf-8")
    result = json.load(f)
    f.close()
    return result;




#######################################################################################################
# Concept-Normalization (C-Norm)
#######################################################################################################

def CNorm(vst_onlyTokens, dl_terms, dl_associations, vso,
          nbEpochs=30, batchSize=64,
          l_numberOfFilters=[4000], l_filterSizes=[1],
          phraseMaxSize=15):

    # Preparing data for SLFNN and S-CNN components:
    dataSCNN, labels, l_unkownTokens, l_uncompleteExpressions = prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize)
    dataSLFNN = numpy.zeros((dataSCNN.shape[0], dataSCNN.shape[2]))
    for i in range( dataSCNN.shape[0]):
        numberOfToken = 0
        for embedding in dataSCNN[i]:
            if not numpy.any(embedding):
                pass
            else:
                numberOfToken += 1
                dataSLFNN[i] += embedding

        if numberOfToken > 0:
            dataSLFNN[i] = dataSLFNN[i] / numberOfToken


    # Input layers:
    inputLP = Input(shape=dataSLFNN.shape[1])
    inputCNN = Input(shape=[dataSCNN.shape[1],dataSCNN.shape[2]])


    # SLFNN component:
    ontoSpaceSize = labels.shape[2]
    denseLP = layers.Dense(units=ontoSpaceSize, use_bias=True, kernel_initializer=initializers.GlorotUniform())(inputLP)
    modelLP = Model(inputs=inputLP, outputs=denseLP)


    # Shallow-CNN component:
    l_subLayers = list()
    for i, filterSize in enumerate(l_filterSizes):

        convLayer = (layers.Conv1D(l_numberOfFilters[i], filterSize, strides=1, kernel_initializer=initializers.GlorotUniform()))(inputCNN)

        outputSize = phraseMaxSize - filterSize + 1
        pool = (layers.MaxPool1D(pool_size=outputSize))(convLayer)

        activationLayer = (layers.LeakyReLU(alpha=0.3))(pool)

        l_subLayers.append(activationLayer)

    if len(l_filterSizes) > 1:
        concatenateLayer = (layers.Concatenate(axis=-1))(l_subLayers)  # axis=-1 // concatenating on the last dimension
    else:
        concatenateLayer = l_subLayers[0]

    denseLayer = layers.Dense(ontoSpaceSize, kernel_initializer=initializers.GlorotUniform())(concatenateLayer)
    modelCNN = Model(inputs=inputCNN, outputs=denseLayer)

    convModel = Model(inputs=inputCNN, outputs=concatenateLayer)
    fullmodel = models.Sequential()
    fullmodel.add(convModel)


    # Combination of the two components:
    combinedLayer = layers.average([modelLP.output, modelCNN.output])
    fullModel = Model(inputs=[inputLP, inputCNN], outputs=combinedLayer)
    fullModel.summary()


    # Compile and train:
    fullModel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=[metrics.CosineSimilarity(), metrics.MeanSquaredError()])
    fullModel.fit([dataSLFNN, dataSCNN], labels, epochs=nbEpochs, batch_size=batchSize)


    return fullModel, vso, l_unkownTokens





#######################################################################################################
# Run class:
#######################################################################################################
class Train(OptionParser):

    def __init__(self):

        OptionParser.__init__(self, usage='usage: %prog [options]')

        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors',
                        help='path to word vectors JSON file as produced by word2vec')
        self.add_option('--word-vectors-bin', action='store', type='string', dest='word_vectors_bin',
                        help='path to word vectors binary file as produced by word2vec')
        self.add_option('--terms', action='store', type='string', dest='terms',
                        help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--attributions', action='store', type='string', dest='attributions',
                        help='path to attributions file in JSON format (map: id -> array of concept ids)')
        self.add_option('--ontology', action='store', type='string', dest='ontology',
                        help='path to ontology file in OBO format')

        self.add_option('--outputModel', action='store', type='string', dest='model', help='path to save the NN model directory')

        # Methods hyperparameters:
        self.add_option('--factor', action='store', type='float', dest='factors', default=0.65, help='parent concept weight factor (default=0.6).')
        self.add_option('--epochs', action='store', type='int', dest='epochs', default=150, help='number of epochs (default=150).')
        self.add_option('--batch', action='store', type='int', dest='batch', default=64, help='number of samples in batch (default=64).')
        self.add_option('--filtersSize', action='append', type='int', dest='filtersSize', help='list of the different size of filters (default=1)')
        self.add_option('--filtersNb', action='append', type='int', dest='filtersNb', help='list of the number of filters from filtersSize (default=100)')
        self.add_option('--phraseMaxSize', action='store', type='int', dest='phrase_max_size', default=15, help='max considered size of phrases in inputs (default=15).')
        self.add_option('--normalizedInputs', action='store', type='string', dest='normalizedInputs', default="True", help='unit normalize embeddings if "True" (default: True).')



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
        if not options.terms:
            raise Exception('missing --terms')
        if not options.attributions:
            raise Exception('missing --attributions')
        if not options.model:
            raise Exception('missing --outputModel')

        if options.filtersSize is None:
            options.filtersSize = [1]
        if options.filtersNb is None:
            options.filtersNb = [100]
        if options.filtersSize is not None and options.filtersNb is not None:
            if len(options.filtersSize) != len(options.filtersNb):
                raise Exception('ERROR: number of elements in --filtersSize different from number of elements in --filtersNb')


        # Selected hyperparameters (can have an important influence...):
        print("\nRuning C-Norm with next hyperparameters:")
        print("factor=", options.factors)
        print("epochs=", options.epochs)
        print("batch=", options.batch)
        print("filtersSize=", options.filtersSize)
        print("filtersNb=", options.filtersNb)
        print("phraseMaxSize=", options.phrase_max_size)
        print("normalizedInputs=", options.normalizedInputs)


        # Loading ontology:
        print("\nloading ontology:", options.ontology)
        ontology = loadOnto(options.ontology)

        # Loading word embeddings:
        if options.word_vectors is not None:
            print("loading word embeddings:", options.word_vectors)
            word_vectors = loadJSON(options.word_vectors)
        elif options.word_vectors_bin is not None:
            print("loading word embeddings:", options.word_vectors_bin)
            EmbModel = gensim.models.Word2Vec.load(options.word_vectors_bin)
            word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in EmbModel.wv[k])) for k in EmbModel.wv.vocab.keys())

        # Loading all mentions:
        print("Loading terms:", options.terms)
        dl_terms = loadJSON(options.terms)

        # Loading training examples:
        print("Loading attributions:", options.attributions)
        attributions = loadJSON(options.attributions)


        # Scaling of all embeddings:
        if options.normalizedInputs is not None:
            if options.normalizedInputs == "True":
                print("\nScaling: Unit normalization of input embeddings (recommended)...")
                word_vectors = normalizeEmbedding(word_vectors)
                print("Scaling done.\n")
        else:
            print("No scaling of input embeddings (not recommended, see normalizedInputs option).")

        # Building ontological space
        print("Building ontological space (with factor applied)...")
        vso = ontoToVec(ontology, options.factors)
        print("Ontological space built.\n")



        print("C-Norm training...")
        model, ontology_vector, _ = CNorm(word_vectors, dl_terms, attributions, vso,
                                nbEpochs=options.epochs, batchSize=options.batch,
                                l_numberOfFilters=options.filtersNb, l_filterSizes=options.filtersSize,
                                phraseMaxSize=options.phrase_max_size)
        print("C-Norm training done.\n")



        # Saving model:
        if options.model is not None:
            print("Saving trained Tensorflow model...")
            d = dirname(options.model)
            if not exists(d) and d != '':
                makedirs(d)
            model.save(options.model, save_format='h5')
            print("Model saved.")



#######################################################################################################
# Test section
#######################################################################################################

if __name__ == '__main__':

    Train().run()
