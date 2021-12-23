""" SECTION 0: IMPORT SECTION"""
# helper functions
from helper_func_and_classes import TwitterDataset_BERT, BERTClassifier
from helper_func_and_classes import create_data_loader_BERT
from helper_func_and_classes import split_dataset
from helper_func_and_classes import create_dataset_list
from helper_func_and_classes import output_numpy_array_from_model_training
from helper_func_and_classes import train_one_epoch, evaluate_model
from helper_func_and_classes import create_predictions_BERT
from helper_func_and_classes import create_submission_file


# transformers
import transformers
from transformers import BertModel, BertTokenizer
from transformers import logging, AdamW, get_linear_schedule_with_warmup

# pytorch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#other
import random
import warnings


RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

# sets up cuda if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup of parameters to create submission csv file
max_length = 37
batch_size = 32
test_size = 0.1
num_classes = 2
dropout_prob = 0.4
num_epoch = 3
learning_rate = 2.5e-5
correct_bias_val = False


""" SECTION 1: Data preprocessing section"""
if __name__ == '__main__':   
    # creating a list of all positive sentences
    pos_data_full = create_dataset_list("./twitter-datasets/train_pos_full.txt")
    pos_labels_full = [1]*len(pos_data_full)

    # creating a list of all negative sentences
    neg_data_full = create_dataset_list("./twitter-datasets/train_neg_full.txt")
    neg_labels_full = [0]*len(neg_data_full)

    all_data_full = pos_data_full + neg_data_full
    all_labels_full = pos_labels_full + neg_labels_full



    # creating a list of sentences that are not yet labeled
    submission_data = create_dataset_list("./twitter-datasets/test_data.txt")
    print("Length of all_data_full: ", len(all_data_full))
    print("Length of all_labels_full: ", len(all_labels_full), "\n")

    print("Length of submission_data: ",len(submission_data))


    PRETRAINED_MODEL_BERT = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_BERT)

    # splits the data into training and testing - using a 90% | 10% split
    train_samples_full, test_samples_full, train_labels_full, test_labels_full = train_test_split(
        all_data_full, 
        all_labels_full, 
        test_size=test_size, 
        random_state=RANDOM_SEED)

    # creates a data loader for the training data
    train_loader_full = create_data_loader_BERT(
        train_samples_full,
        train_labels_full,
        tokenizer, 
        max_length, 
        batch_size)

    # creates a data loader for the testing data
    test_loader_full = create_data_loader_BERT(
        test_samples_full,
        test_labels_full,
        tokenizer, 
        max_length, 
        batch_size)


    """ SECTION 1: Data preprocessing section"""
    # loading in the pretrained model
    model_bert = BertModel.from_pretrained(PRETRAINED_MODEL_BERT, return_dict=False)

    # initializing the BERT classifier model
    model = BERTClassifier(
        num_classes=num_classes,
        p=dropout_prob,
        pretrained_model_name=PRETRAINED_MODEL_BERT)

    # setting the model to gpu (if possible)
    model = model.to(device)


    # calculating the total number of steps for the data
    total_steps = len(train_loader_full)*num_epoch

    optimizer = AdamW(model.parameters(), correct_bias=correct_bias_val, lr=learning_rate)

    # setting up a scheduler with 30 000 warmup steps
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=30000,
    num_training_steps=total_steps
    )

    # Cross Entropy Loss 
    loss_func = nn.CrossEntropyLoss()

    # setting up empty lists for plotting
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    # actual training of data
    for epoch in range(num_epoch):
        print("Epoch: ", int(epoch+1))
        
        # does one epoch of training
        train_accuracy, loss = train_one_epoch(
            model,
            train_loader_full,
            loss_func,
            optimizer,
            device,
            scheduler,
            len(train_samples_full)
            )

        # evaluates the current state of the model
        test_accuracy, test_loss = evaluate_model(
            model,
            test_loader_full,
            device,
            loss_func,
            len(test_samples_full)
            )

        print(f'Train loss {loss} accuracy {train_accuracy}')
        print(f'Test loss {test_loss} accuracy {test_accuracy}\n')

        # appending data to lists
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)


    # submission data loader
    submission_ids = []
    for i in range(10000):
        submission_ids.append(i+1)
        
    submission_loader = create_data_loader_BERT(
        submission_data,
        submission_ids, 
        tokenizer,
        max_length, 
        batch_size)

    # creating predicitons based on the submission data, will output in numpy format
    numpy_predictions = create_predictions_BERT(model, submission_loader, device)

    # will go trough the numpy array and writing -1 and 1 to file
    # 0 in numpy array will give -1 csv
    # 1 in numpy array wil give +1 or 1 in csv
    create_submission_file(numpy_predictions)