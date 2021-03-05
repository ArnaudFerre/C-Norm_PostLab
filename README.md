**# C-Norm (Concept NORMalization)**


The entity normalization method implemented in this repository is an optimization of the C-Norm method and is implemented on the [PostLab](https://www.postlab.fr/) platform, and is usable with graphical user interface (work in progress).<br />
<br />
The initial method is described in the article below:<br />
C-Norm: a Neural Approach to Few-Shot Entity Normalization<br />
Arnaud Ferré-1, Louise Deléger-1, Robert Bossy-1, Pierre Zweigenbaum-2, Claire Nédellec-1<br />
1-Université Paris-Saclay, INRAE MaIAGE, Jouy-en-Josas, France<br />
2-Université Paris-Saclay, CNRS LIMSI, Orsay, France<br />
Web link: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03886-8<br />
If you want to reproduce the different experiments described in this article, see rather the initial repository of C-Norm with its sidekick methods: https://github.com/ArnaudFerre/C-Norm/blob/master/README.md<br />
<br />
This code is under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

<br />

**Command lines for testing:**

C-Norm training on tool dataset:
`python train.py --word-vectors-bin data/VST_count0_size100_iter50.model --ontology data/OntoBiotope_BioNLP-ST-2016.obo --terms data/terms_train.json --attributions data/attributions_train.json --outputModel data/trainedTFmodel/ --epochs 5`

C-Norm prediction on tool dataset (on dev set):<br/>
`python predict.py --word-vectors-bin data/VST_count0_size100_iter50.model --ontology data/OntoBiotope_BioNLP-ST-2016.obo --terms data/terms_dev.json --inputModel data/trainedTFmodel/ --output data/results/testResults.txt`

<br />

**Information on possible parameters:**<br />

For main_train.py, mandatory parameters:<br />
--word-vectors-bin: path to the embeddings file of tokens, Gensim model format. If you have no Gensim model, you can load in JSON format with option --word-vectors instead: `{"token1": [value11, …, value1N], "token2": [value21, …, value2N], …}`<br />
--ontology: path to the ontology file (OBO format, some OWL format) used to normalize mentions (i.e the identifiers of the concepts must be the same used in the attributions file).<br />
--terms: path to the JSON file containing the mentions from examples, with their word segmentation. Format: `{"mention_unique_id1": ["token11", "token12", …, "token1m"], "mention_unique_id2": ["token21", …, "token2n"], … }`<br />
--attributions: path to the JSON file with the attributions of concept(s) to each mention for the training. Format: `{"mention_unique_idA":["concept_identifierA1", "concept_identifierA2", …], "mention_unique_idB":["concept_identifierB1"], …}`<br />
--outputModel: path to a directory where save the training parameters (i.e. Tensorflow model).<br />
<br />
For main_train.py, optional parameters:<br />
--phraseMaxSize: Integer. This number set the max tokens taking into account in mentions (default=15).<br />
--normalizedInputs: True or False. A possible scaling of embeddings. Best results with unit normalization (True value by default).<br />
--factor: value to smooth the concept vectors weights. If factor=0, concept vectors are one-hot encoding. If factor=1, for each concept vector, the weight associated to its parent concept is equal to 1. Good value is factor=0,6 (=default value).<br />
--epochs: number of time that the program will be train on the same training data. Try different values. In our experiments, best results with a value between 30 and 200 (default value=150).<br />
--batch: number of training examples seen at the same time by the program (default value=64).<br />
--filtersSize: list of integers. Each value represents the width of a typ of filter applied in the CNN part of C-Norm (default value=1).<br />
--filtersNb: list of integers. Must have same number of value than filtersSize. Each value represents the number of filters respectively for each filter type (default value=100, really not optimal, but quick to run).<br />
<br />
For predict.py, mandatory parameters:<br />
--word-vectors-bin: path to the embeddings file of tokens, Gensim model (or JSON with --word-ectors option). Use the same embeddings that in your training set.<br />
--ontology: path to the ontology file (OBO format, some OWL format) used to normalize mentions. Use the same ontology file that in your training set.<br />
--terms: path to the JSON file containing the mentions from examples, with their word segmentation. Format: `{"mention_unique_id1": ["token11", "token12", …, "token1m"], "mention_unique_id2": ["token21", … "token2n"], … }`<br />
--inputModel: path where is located the Tensorflow model after training.<br />
--output: filepath where save the predictions. CSV format: `mention_id concept_id	similarity_value`<br />
<br />
For predict.py, optional parameters:<br />
--phraseMaxSize: Must be the same value that for the training (default=15).<br />
--normalizedInputs: Recommended to use the same value than for the training.<br />
--factor: value to smooth the concept vectors weights. Preferentially, use the value that in your training set.<br />

<br />

**Dependencies:**<br /><br />
Language: Python 3.6.9<br />
<br />
Python libraries:<br />
--[Tensorflow 2.0.0](https://www.tensorflow.org/install) : neural networks<br />
--[Pronto](https://pypi.org/project/pronto/): ontology manipulation (OBO, possibly OWL)<br /> 
--[Gensim](https://radimrehurek.com/gensim/models/word2vec.html): creation/manipulation of word embeddings (optional - JSON equivalent format normally tolerated)<br />
--[Numpy](https://numpy.org/)<br />
--[scikit-learn](https://scikit-learn.org/stable/install.html)<br />
<br />
External tools (managed by the [AlvisNLP/ML](https://bibliome.github.io/alvisnlp/) suite in the published experiments):<br />
--Sentence/word segmenter : [SeSMig](https://bibliome.github.io/alvisnlp/reference/module/SeSMig) / [WoSMig](https://bibliome.github.io/alvisnlp/reference/module/WoSMig) <br />
--Named entity recognition: needed by the method first (C-Norm retrieves the results of a recognition as input)<br />
