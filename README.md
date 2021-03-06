


# Replication of "What Part of the Neural Network Does This"

We're working on a replication of results from the following paper:

> Xin, J., Lin, J., & Yu, Y. (2019, November). What Part of the Neural Network Does This? Understanding LSTMs by Measuring and Dissecting Neurons. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 5827-5834).

* [1. Usage](#Usage)
* [2. Findings](#Findings)
* [3. About the Framework](#About-the-Framework-We-used)

# Usage
We added code for calculating sensitivity, importance rankings, accuracies, ablating neurons, correlations, similarity, overlap, as well as preprocessing the dataset.
* [1. Train](#Train-and-test)
* [2. Sensitivities](#Sensitivities)
* [3. Importance Rankings](#Importance-Rankings)
* [4. Accuracies](#Accuracies)
* [5. Ablating neurons](#Ablating-neurons)
* [6. Correlations](#Correlations)
* [7. Similarity](#Similarity)
* [8. Overlap](#Overlap)
* [9. Preprocessing](#preprocessing)

## Train and test
> **Note:** A prerequisite for training the model is the GloVe embeddings. You need to get them from
> https://nlp.stanford.edu/projects/glove/ and put `glove.6B.zip` in the directory `test_data`. Then you can set
> `word_emb_dir` in `test.train.config` to point to `test_data/glove.6B.zip/glove.6B.50d.txt` to use 50-dimensional embeddings, or 
> `test_data/glove.6B.zip/glove.6B.100d.txt` to use 100-dimensional embeddings.

**Train**: Training the model from the paper takes about 10 minutes. To train a model, edit a config file (an example is test.train.config), so that `model_dir` indicating the path and beginning of the filename to where you want to save your trained model. For example, `model_dir=test_data/lstmtest50` will save the model in the file `test_data/lstmtest50.9.model` (the 9 is for the 10th epoch of training starting at index 0 so that's why it's 9 and not 10). The config file consumes some preprocessed data we put in the `test_data` directory.

Then run:
> python3 main.py --config test.train.config 

**Test**: To run some of the evaluations that get data for our charts, run (change --pretrainedmodelpath to match what you set in the config file, plus ".9.model".:
> python3 main.py --config test.train.config --loadtotest True --pretrainedmodelpath "test_data/lstmtest50.9.model" --ablate 0

## Sensitivities
**Sensitivity heatmap** is generated in the `/test_data` directory (or whereever you specified in the config file) and has a name like `lstmtest50.9.model_heatmap.png`. There's also a `lstmtest50.9.model_heatmap.npy` used to calculate correlations between models. Here is a sensitivity heatmap for 50 neurons and nine BIO labels. It was generated in `heatmap_sensitivity()` in `main.py`.
![Example heatmap](readme/heatmap.png)

If you change the random seed (find `seed_num = 42` in `main.py`, change it and train another model), you'll get a different heatmap. Here's the heatmap for a model with the same parameters but a different random seed as the previous one:
Notice that it's different, and the range of values may be different. But there are some similarities too.
![Example heatmap](readme/heatmap2.png)


## Importance rankings
**Importance rankings** for neurons are generated in files `ImportanceRankings.png`, `Importance.txt`, `Importance.tsv`, and `imps.npy`. The last one is used to calculate [*overlap*](#overlap). 
Using the sensitivity matrix shown in the heatmap, we determine the importance ranking of the each neuron and list them from most to least important. Here an example of the `ImportanceRankings.png`.

![Importance ranking](readme/importance.png)

## Accuracies
**Accuracies** To save accuracies to files or show accuracies in the console output, set the parameters in the following functions to **True**.
    def get_per_tag_accuracy(expected_tag_counts, actual_tag_counts, sort=False, data=None,
        write_file=False, print_accs=False)
	
Accuracies are written to files in the root directory with names like `TAGn_acc.txt` for accuracy when you ablate n neurons. 

We also generate charts similar to one below. An example command line with the `--ablate` argument to generate charts of ablating 10 neurons of each of the tags:
`python main.py --config test.train.glove50.config --loadtotest True --pretrainedmodelpath "test_data/lstmtestgloveB50.9.model" --ablate 10`

![comparing embeddings](readme/B-ORG-embedding-compare.png) 
The chart shows how accuracy degrades over when you ablate important neurons. Our automatically generated charts are saved in the root directory with names like `B-ORG_chart.png`.

The axes of the heatmaps list the NER tags that the model was trying to label. The accuracy rates for each tag vary, for example they might look like:

	B-LOC	0.9069134458356015
	B-MISC	0.702819956616052
	B-ORG	0.7225950782997763
	B-PER	0.9310532030401737
	I-LOC	0.669260700389105
	I-MISC	0.5549132947976878
	I-ORG	0.6125166444740346
	I-PER	0.9563886763580719
	O	0.995927770279704

## Ablating neurons
**Ablating neurons** The `--ablate` flag specifies how many neurons to ablate. You get the list of neurons to ablate from the importance ranking files or the console output, for a specified tag like B-ORG or I-MISC. Paste this list into `forward()` in `wordsequence.py` where we have a comment about `"Ablation of neurons"` for the value of `feature order`, and then run a command like the following which specifies that you want to ablate the top ten neurons.

> python3 main.py --config test.train.config --loadtotest True --pretrainedmodelpath "test_data/lstmtest50.9.model" --ablate 10

## Correlations
To calculate the correlations between the neuron sensitivitities for models with different random seeds, you can use the code in `utils/corr.py` or in `correlation_plotting.ipynb`. This will save the correlation heatmap to '/test_data/lstmtestglove50.9.model_sensitivities_correlation.png'. You can change that path to whereever you saved the trained model. The following image is the sensitivity correlation matrix for one of the models we tested:

![sensitivity_correlation](readme/lstmtestglove50.9.model_sensitivities_correlation.png)

We describe what the correlations show in [Findings](#Findings).

## Similarity
In an experiment beyond what the paper did, we measure the cosine similarity between learned weights for a pair of labels in the fully-connected layer of the model to see if there's correlation between models with different random seeds and with ablation patterns. The code for this is in `utils/weight-similarity.py`. The weights this function uses are saved in `weights.npy` after you train the model.

![Similarity](readme/Similarity_correlation.png)

## Overlap
In an experiment beyond what the paper did, we measure the shared neurons in the top-ten most-important neurons of a pair of labels in the model to see if there's correlation between models with different random seeds and with ablation patterns. The code for this is in `overlap()` in `utils/weight-similarity.py`. An example of the values for different pairs of labels is shown here in the `Importance_overlap_top_ten.png` file that we generate:
![Importance overlap](readme/Importance_overlap_top_ten.png)

## Preprocessing additional datasets
The preprocessing code in `preprocessing.ipynb` could be used for additional datasets. We processed some additional datasets which are in subfolders of the `data` directory. Our future plans include running experiments on these datasets.

# Findings
We observe the following results for **sensitivity** correlation that are consistent with the original paper:
* Label pairs of the form B-x and I-x (where x
is PER/LOC/ORG/MISC) are generally positively
correlated. We can observe some darkred
2 2 blocks on the diagonal. Although for
each trained model, it might be different neurons
(i.e., neuron #) that encode information
about B-x, these neurons typically also carry
information about I-x.
* The label triples I-LOC, I-ORG, and I-MISC
are also positively correlated.
* Label pairs of the form B-x and I-y (where
x and y are different entities) are generally
negatively correlated.
* The label O is negatively correlated with all
I-x labels.

We observe the following results for **similarity** between labels based on the vector of learned weights in the fully-connected layer:
* Label pairs of the form B-x and I-x (where x
is PER/LOC/ORG/MISC) are **not** the most similar. 
* The label triples I-LOC, I-ORG, and I-MISC
are similar.
* Labels of the form B-x are more similar to labels B-y than to label I-x (where
x and y are different entities), except in the case of B-PER and I-PER, which are similar.
* The label O is dissimilar to most labels but the closest to I-MISC.

# About the framework we used

Most of the codebase is from NCRF++: An Open-source Neural Sequence Labeling Toolkit. The reference for it follows.
<!-- **Note:** Not planning to leave the whole NCRF++ reference here, but just for now to use as a guide for our own readme. -->

# Reference for NCRF++: An Open-source Neural Sequence Labeling Toolkit
* [1. Introduction](#Introduction)
* [2. Requirement](#Requirement)
* [3. Advantages](#Advantages)
* [4. Usage](#Usage)
* [5. Data Format](#Data-Format)
* [6. Performance](#Performance)
* [7. Add Handcrafted Features](#Add-Handcrafted-Features)
* [8. Speed](#Speed)
* [9. N best Decoding](#N-best-Decoding)
* [10. Reproduce Paper Results and Hyperparameter Tuning](#Reproduce-Paper-Results-and-Hyperparameter-Tuning)
* [11. Report Issue or Problem](#Report-Issue-or-Problem)
* [12. Cite](#Cite)
* [13. Future Plan](#Future-Plan)
* [13. Update](#Update)

## Introduction

Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation. State-of-the-art sequence labeling models mostly utilize the CRF structure with input word features. LSTM (or bidirectional LSTM) is a popular deep learning based feature extractor in sequence labeling task. And CNN can also be used due to faster computation. Besides, features within word are also useful to represent word, which can be captured by character LSTM or character CNN structure or human-defined neural features.

NCRF++ is a PyTorch based framework with flexiable choices of input features and output structures. The design of neural sequence labeling models with NCRF++ is fully configurable through a configuration file, which does not require any code work. NCRF++ can be regarded as a neural network version of [CRF++](http://taku910.github.io/crfpp/), which is a famous statistical CRF framework. 

This framework has been accepted by [ACL 2018](https://arxiv.org/abs/1806.05626) as demonstration paper. And the detailed experiment report and analysis using NCRF++ has been accepted at [COLING 2018](https://arxiv.org/abs/1806.04470) as the best paper.

NCRF++ supports different structure combinations of on three levels: character sequence representation, word sequence representation and inference layer.

* Character sequence representation: character LSTM, character GRU, character CNN and handcrafted word features.
* Word sequence representation: word LSTM, word GRU, word CNN.
* Inference layer: Softmax, CRF.

Welcome to star this repository!

## Requirement

	Python: 2 or 3  
	PyTorch: 1.0 

[PyTorch 0.3 compatible version is here.](https://github.com/jiesutd/NCRFpp/tree/PyTorch0.3)


## Advantages

* Fully configurable: all the neural model structures can be set with a configuration file.
* State-of-the-art system performance: models build on NCRF++ can give comparable or better results compared with state-of-the-art models.
* Flexible with features: user can define their own features and pretrained feature embeddings.
* Fast running speed: NCRF++ utilizes fully batched operations, making the system efficient with the help of GPU (>1000sent/s for training and >2000sents/s for decoding).
* N best output: NCRF++ support `nbest` decoding (with their probabilities).


## Usage

NCRF++ supports designing the neural network structure through a configuration file. The program can run in two status; ***training*** and ***decoding***. (sample configuration and data have been included in this repository)  

In ***training*** status:
`python main.py --config demo.train.config`

In ***decoding*** status:
`python main.py --config demo.decode.config`

The configuration file controls the network structure, I/O, training setting and hyperparameters. 

***Detail configurations and explanations are listed [here](readme/Configuration.md).***

NCRF++ is designed in three layers (shown below): character sequence layer; word sequence layer and inference layer. By using the configuration file, most of the state-of-the-art models can be easily replicated ***without coding***. On the other hand, users can extend each layer by designing their own modules (for example, they may want to design their own neural structures other than CNN/LSTM/GRU). Our layer-wised design makes the module extension convenient, the instruction of module extension can be found [here](readme/Extension.md).

![alt text](readme/architecture.png "Layer-size design")


## Data Format

* You can refer the data format in [sample_data](sample_data). 
* NCRF++ supports both BIO and BIOES(BMES) tag scheme.  
* Notice that IOB format (***different*** from BIO) is currently not supported, because this tag scheme is old and works worse than other schemes [Reimers and Gurevych, 2017](https://arxiv.org/pdf/1707.06799.pdf). 
* The difference among these three tag schemes is explained in this [paper](https://arxiv.org/pdf/1707.06799.pdf).
* I have written a [script](utils/tagSchemeConverter.py) which converts the tag scheme among IOB/BIO/BIOES. Welcome to have a try. 


## Performance

Results on CONLL 2003 English NER task are better or comparable with SOTA results with the same structures. 

CharLSTM+WordLSTM+CRF: 91.20 vs 90.94 of [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf);

CharCNN+WordLSTM+CRF:  91.35 vs 91.21 of [Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).   

By default, `LSTM` is bidirectional LSTM.    

|ID| Model | Nochar | CharLSTM |CharCNN   
|---|--------- | --- | --- | ------    
|1| WordLSTM | 88.57 | 90.84 | 90.73  
|2| WordLSTM+CRF | 89.45 | **91.20** | **91.35** 
|3| WordCNN |  88.56| 90.46 | 90.30  
|4| WordCNN+CRF |  88.90 | 90.70 | 90.43  

We have compared twelve neural sequence labeling models (`{charLSTM, charCNN, None} x {wordLSTM, wordCNN} x {softmax, CRF}`) on three benchmarks (POS, Chunking, NER) under statistical experiments, detail results and comparisons can be found in our COLING 2018 paper [Design Challenges and Misconceptions in Neural Sequence Labeling](https://arxiv.org/abs/1806.04470).
 

## Add Handcrafted Features

NCRF++ has integrated several SOTA neural characrter sequence feature extractors: CNN ([Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)), LSTM ([Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)) and GRU ([Yang .etc, ICLR17](https://arxiv.org/pdf/1703.06345.pdf)). In addition, handcrafted features have been proven important in sequence labeling tasks. NCRF++ allows users designing their own features such as Capitalization, POS tag or any other features (grey circles in above figure). Users can configure the self-defined features through configuration file (feature embedding size, pretrained feature embeddings .etc). The sample input data format is given at [train.cappos.bmes](sample_data/train.cappos.bmes), which includes two human-defined features `[POS]` and `[Cap]`. (`[POS]` and `[Cap]` are two examples, you can give your feature any name you want, just follow the format `[xx]` and configure the feature with the same name in configuration file.)
User can configure each feature in configuration file by using 

```Python
feature=[POS] emb_size=20 emb_dir=%your_pretrained_POS_embedding
feature=[Cap] emb_size=20 emb_dir=%your_pretrained_Cap_embedding
```

Feature without pretrained embedding will be randomly initialized.


## Speed

NCRF++ is implemented using fully batched calculation, making it quite effcient on both model training and decoding. With the help of GPU (Nvidia GTX 1080) and large batch size, LSTMCRF model built with NCRF++ can reach 1000 sents/s and 2000sents/s on training and decoding status, respectively.

![alt text](readme/speed.png "System speed on NER data")


## N best Decoding

Traditional CRF structure decodes only one label sequence with largest probabolities (i.e. 1-best output). While NCRF++ can give a large choice, it can decode `n` label sequences with the top `n` probabilities (i.e. n-best output). The nbest decodeing has been supported by several popular **statistical** CRF framework. However to the best of our knowledge, NCRF++ is the only and the first toolkit which support nbest decoding in **neural** CRF models. 

In our implementation, when the nbest=10, CharCNN+WordLSTM+CRF model built in NCRF++ can give 97.47% oracle F1-value (F1 = 91.35% when nbest=1) on CoNLL 2003 NER task.

![alt text](readme/nbest.png  "N best decoding oracle result")


## Reproduce Paper Results and Hyperparameter Tuning

To reproduce the results in our COLING 2018 paper, you only need to set the `iteration=1` as `iteration=100` in configuration file `demo.train.config` and configure your file directory in this configuration file. The default configuration file describes the `Char CNN + Word LSTM + CRF` model, you can build your own model by modifing the configuration accordingly. The parameters in this demo configuration file are the same in our paper. (Notice the `Word CNN` related models need slightly different parameters, details can be found in our COLING paper.)

If you want to use this framework in new tasks or datasets, here are some tuning [tips](readme/hyperparameter_tuning.md) by @Victor0118.


## Report Issue or Problem

If you want to report an issue or ask a problem, please attach the following materials if necessary. With these information, I can give fast and accurate discussion and suggestion. 
* `log file` 
* `config file` 
* `sample data` 


## Cite

If you use NCRF++ in your paper, please cite our [ACL demo paper](https://arxiv.org/abs/1806.05626):

    @inproceedings{yang2018ncrf,  
     title={NCRF++: An Open-source Neural Sequence Labeling Toolkit},  
     author={Yang, Jie and Zhang, Yue},  
     booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
     Url = {http://aclweb.org/anthology/P18-4013},
     year={2018}  
    }


If you use experiments results and analysis of NCRF++, please cite our [COLING paper](https://arxiv.org/abs/1806.04470):

    @inproceedings{yang2018design,  
     title={Design Challenges and Misconceptions in Neural Sequence Labeling},  
     author={Yang, Jie and Liang, Shuailong and Zhang, Yue},  
     booktitle={Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
     Url = {http://aclweb.org/anthology/C18-1327},
     year={2018}  
    }

## Future Plan 

* Document classification (working)
* Support API usage
* Upload trained model on Word Segmentation/POS tagging/NER
* Enable loading pretrained ELMo parameters
* Add BERT feature extraction layer 



## Update

* 2018-Dec-17, NCRF++ v0.2, support PyTorch 1.0
* 2018-Mar-30, NCRF++ v0.1, initial version
* 2018-Jan-06, add result comparison.
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version

