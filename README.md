![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
# CS-433 Project 2, 2021, EPFL Text classification
## Author: 
- Fridtjof Storm Flaate | fridtjof.flaate@epfl.ch
## Submission:
**Submission #169334**
## Abstract:
The goal of this project was to build a model that could accurately classify tweets as either positive or negative. In this project, you will find six different models. Three classic machine learning models and three neural networks. The best performing model is the neural network using the pre-trained bert-base-casebidirectional encoder representations from transformers, also called BERT. The transfer-learning model gave us an accuracy of 89.2% and an F1 score of 89.5%.


## Setup:
This is a step by step guide of how you can setup up your environment to run the run.py that will create the submission file.

### Prerequisites

* conda
* pip3
* python3
* Download 'epfml-text' from [here](https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files), unzip and add to `/twitter-datasets` folder.


### Installation
1. Clone the repo and enter directory `text_classification`
   ```sh
   git clone https://github.com/StormFlaate/text_classification 
   ```
2. create environment
   ```sh
   conda create --name text_classification
   ```
3. activate environment
   ```sh
   conda activate text_classification
   ```
4. install dependencies
   ```sh
   conda install --file requirements.txt && conda install -c huggingface transformers

   ```
5. install dependencies
   ```sh
   pip3 install -r requirements_pip.txt
   ```

## Overview:
### Setup files:
- [requirements.txt](./requirements.txt): file to install conda dependencies  
- [requirements_pip.txt](./requirements_pip.txt): file to install python3 dependecies
- [README.md](./README.md): file containing information about the project<br>
### Machine learning models:
- [SGD log loss](./SGD_classifier.ipynb): trianing and testing of model
- [Logisitic regression](./LogisticRegression_classification.ipynb): training and testing of model
- [Random forest](.RandomForest_classifier.ipynb): training and testing of model
- [NN GloVe](./NeuralNetworkGloVe.ipynb): Neural network with aggregated GloVe word embeddings
- [NN sentence transformer](./NeuralNetwork_MiniLM.ipynb): NN with all-MiniLM-L6-v2 sentence embedding
- [NN transfer learning BERT](./BERT_classification.ipynb): Transfer learning model BERT
### Run files:
- [run.py](./run.py): file containing everything to recreate best submission
### Helper functions:
- [helper functions](./helper_func_and_classes.py): contains all helper functions and classes used in the project
### Folders:
- [twitter-datasets](./twitter-datasets): will contain all data-sets used for this project - need to be downlaoded manually.
