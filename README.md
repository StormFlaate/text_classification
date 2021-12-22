# CS-433 Project 2, 2021, EPFL Text classification
## Author: 
- Fridtjof Storm Flaate

## Abstract:
The goal of this project was to build a model that could accurately classify tweets as either positive or negative. In this project, you will find six different models. Three classic machine learning models and three neural networks. The best performing model is the neural network using the pre-trained bert-base-casebidirectional encoder representations from transformers, also called BERT. The transfer-learning model gave us an accuracy of $xx.xx\%$ and an F1 score of $xx.xx\%$.


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
   conda install --file requirements.txt
   ```
5. install dependencies
   ```sh
   pip3 install -r requirements_pip.txt
   ```

