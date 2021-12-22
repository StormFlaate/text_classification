# pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import normalize

from torch.utils.data import Dataset, DataLoader, TensorDataset


# transformers
import transformers
from transformers import BertModel


# NLP helpers
from nltk.stem import PorterStemmer

#scikit-learn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import scale
# data science
import numpy as np

#python standard library
import random
from random import shuffle
from math import ceil
import csv


# helpers
from tqdm import tqdm
RANDOM_SEED = 123


def create_dataset_list(file_path_pos):
    """
    Explanation:        Takes in one filepath, creates list with one sentence in each
        
    INPUT:
     - file_path_pos:   file path to the positive tweets      
     - file_path_neg:   file path to the negative tweets
    
    RETURN:
     - data:             list containing all sentences in the file right-stripped
    """
    data = list(map(str.rstrip, open(file_path_pos)))
    return data


def create_submission_file(label_prediction):
    """
    Explanation:            Takes in labels in the correct order and create a submission file (.csv)
    
    INPUT:
     - label_prediction:    array containing zeros and ones, length is 10 000

    OUTPUT:
     - void:                creates a csv file containing Id, Prediction (10 000 rows) with -1 and 1
    """
    f = open('./submission_labels.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Id', 'Prediction'])
    for index, label in enumerate(label_prediction):
        row = [index+1,0]
        if label == 0:
            row[1] = -1
        else:
            row[1] = 1
        writer.writerow(row)
    f.close()


def create_vocab(file_path_pos, file_path_neg, file_path_test, stem=False):
    """
    Explanation:        Translates negative tweets, positive tweets and test tweets into a lookup dictionary
        
    INPUT:
     - file_path_pos:   file path to the positive tweets      
     - file_path_neg:   file path to the negative tweets
     - file_path_test:  file path to the test tweets
     - stem:            if the words should be stemmed or not (default to false)
    
    RETURN:
     - vocab_dict:      dictionary that contains key:word -> value:index 
    """

    ps = PorterStemmer()

    # reading lines from the positive tweets, right-stripping lines
    raw_pos_list = create_dataset_list(file_path_pos)
    pos_list = []
    
    # creating a set of unique words from the positive tweets
    for line in tqdm(raw_pos_list):
        new_line = line.split(" ")
        if stem:
            new_line = [ps.stem(word) for word in new_line]
        pos_list.extend(new_line)
    pos_set = set(pos_list)
    
    # reading lines from the negative tweets, right-stripping lines
    raw_neg_list = create_dataset_list(file_path_neg)
    neg_list = []
    
    # creating a set of unique words from the positive tweets
    for line in tqdm(raw_neg_list):
        new_line = line.split(" ")
        if stem:
            new_line = [ps.stem(word) for word in new_line]
        neg_list.extend(new_line)
    neg_set = set(neg_list)
    
    # reading lines from the negative tweets, right-stripping lines
    raw_test_list = create_dataset_list(file_path_test)
    test_list = []
    
    # creating a set of unique words from the test set
    for line in tqdm(raw_test_list):
        new_line = line.split(" ")
        if stem:
            new_line = [ps.stem(word) for word in new_line]
        test_list.extend(new_line)
    test_set = set(test_list)
    
    # creates a set of unique words from both neg, pos and test words
    pos_set |= neg_set
    pos_set |= test_set

    sorted_full_list = sorted(list(pos_set))
    
    # creates a lookup dictionary for the words
    vocab_dict = {}
    for index, word in enumerate(sorted_full_list):
        vocab_dict[word] = index
    
    return vocab_dict


class TwitterDataset(Dataset):
    """
    Explanation:        Twitter dataset loader
    
    FUNCTIONS:
     - __init__:        setup, creates dataset and labels
     - __getitem__:     returns a tuple, (single tweet in list format, label to tweet)
     - __len__:         returns length of datset
     - vocab_len:       returns length of vocabulary
    
    """
    def __init__(self, word_to_index_dct, data_pos, data_neg, data_submission, max_sentence_length):
        self.dataset = []
        self.text_vocab = word_to_index_dct
        self.labels = []
        self.submission_dataset = []
        
        # positive data
        for sentence in tqdm(data_pos):
            # tokens of sentence: "Hello my name" -> ["Hello", "my", "name"]
            tokens = sentence.split(" ")
            # Removing outlier tweets to make maximum length of sentence vector shorter
            if len(tokens) < max_sentence_length:
                self.dataset.append([self.text_vocab[word] for word in tokens])
                self.labels.append(1)
        
        # negative data
        for sentence in tqdm(data_neg):
            # tokens of sentence: "Hello my name" -> ["Hello", "my", "name"]
            tokens = sentence.split(" ")
            # Removing outlier tweets to make maximum length of sentence vector shorter
            if len(tokens) < max_sentence_length:
                self.dataset.append([self.text_vocab[word] for word in tokens])
                self.labels.append(0)
                
        for id_, sentence in enumerate(data_submission):
            tokens = sentence.split(" ")
            self.submission_dataset.append([[self.text_vocab[word] for word in tokens], id_])
        
    def __getitem__(self, index):
        # returns a sentence and its corresponding label
        item = self.dataset[index]
        label = self.labels[index]
        return item, label

    def __len__(self):
        return len(self.dataset) 
    
    def vocab_len(self):
        return len(self.text_vocab)


class TwitterDataset_BERT(Dataset):
    """
    Explanation:        Twitter dataset loader, specific for BERT
    
    FUNCTIONS:
     - __init__:        setup, loads in dataset, labels, tokenizer and fixed size of sentences
     - __getitem__:     returns a dicitionary containing input_ids, attention_mask and labels. Format: torch.tensor
     - __len__:         returns length of dataset
    
    """
    def __init__(self, samples, labels, tokenizer, max_len):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = str(self.samples[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
        sample,
        add_special_tokens=True,
        max_length=self.max_len,
        return_attention_mask=True,
        pad_to_max_length=True,
        return_token_type_ids=False,
        return_tensors='pt')

        return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    """
    Explanation:                This is a neural network architecture which builds on top of the pretrained BertModel, 
                                the last hidden linear layer is a dimension = 768. Use dropout for regularization
    
    INPUT: 
     - num_classes:             number of different classes we are predicting
     - p:                       dropout probability
     - pretrained_model_name:   name of the bert model (for example: 'bert-base-uncased')

    OUTPUT:
     - x:                       outputs a raw value for each class
    """

    # Initializes the classifier with correct parameters
    def __init__(self, num_classes, p, pretrained_model_name):
        super(BERTClassifier, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(pretrained_model_name, return_dict=False)
        self.fc1 = nn.Linear(self.bert_pretrained.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(p=p)
        
    # feed forward function in the neural network
    def forward(self, input_ids, attention_mask):
        _, pooled_x = self.bert_pretrained(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        x = self.dropout(pooled_x)
        x = self.fc1(x)
        return x


class NN(nn.Module):
    """
    EXPLANATION:  Class for neural network architecture using 2 hidden layers with 200 nodes in each
    """
    def __init__(self, input_size, num_classes, p):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc9 = nn.Linear(200, num_classes)
        self.dropout1 = nn.Dropout(p) 
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(200)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.fc9(x)
        
        
        return x 


class NN_se(nn.Module):
    """
    EXPLANATION:  Class for neural network architecture using 2 hidden layers with 384 nodes in each
    """
    def __init__(self, input_size, num_classes, p):
        super(NN_se, self).__init__()
        self.fc1 = nn.Linear(input_size, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc9 = nn.Linear(384, num_classes)
        self.dropout1 = nn.Dropout(p) 
        self.batchnorm1 = nn.BatchNorm1d(384)
        self.batchnorm2 = nn.BatchNorm1d(384)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.fc9(x)
        
        return x 


def train_one_epoch(model, data_loader, loss_func, optimizer, device, scheduler, num_datapoints):
    """
    Explanation:          This function will give each sample in the training data an opportunity to update the model
    
    INPUT: 
     - model:             current state of the model
     - date_loader:       the data_loader that will be used on the current epoch, contains both samples and labels
     - loss_func:         the loss function that will be used on the current epoch
     - optimizer:         the optimizer used for the model, used to change weights to reduce loss
     - device:            device for gpu cuda
     - scheduler:         the scheduler used for the model, used to update learning rate
     - num_datapoints:    the total number of samples in current epoch, used to calculate accuracy

    OUTPUT:
    - accuracy:           accuracy of model at current epoch
    - mean_loss:          mean loss of model at curren epoch 
    """

    model = model.train()
    losses = []
    correct_predictions = 0
    for subpart in tqdm(data_loader):
        
        input_ids = subpart["input_ids"].to(device) 
        attention_mask = subpart["attention_mask"].to(device)
        labels = subpart["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, predicted = torch.max(outputs, dim=1)
        
        # calculating loss
        loss = loss_func(outputs, labels)
        correct_predictions += torch.sum(predicted == labels)
        
        # current batch loss
        losses.append(loss.item())

        # does backward pass, computing gradient for every parameter
        loss.backward()
        # computes norm over all gradients together
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # resetting gradients so we don't accumulate the gradients after several passes
        optimizer.zero_grad()
    
    mean_loss = np.mean(losses)
    accuracy = correct_predictions.double() / num_datapoints
    return accuracy, mean_loss


def evaluate_model(model, data_loader, device, loss_func, num_datapoints):
    """
    Explanation:          This function will evaluate the model -> give accuracy and mean loss
    
    INPUT: 
     - model:             current state of the model
     - date_loader:       the data_loader that will be used, contains both samples and labels
     - device:            device for gpu cuda
     - loss_func:         the loss function that will be used
     - num_datapoints:    the total number of samples, used to calculate accuracy

    OUTPUT:
    - accuracy:           accuracy of model 
    - mean_loss:          mean loss of model 
    """
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    # turn off gradients calculation so we don't update model while running test data through model
    with torch.no_grad():
        # loads in batches of data form the data_loader
        for subpart in data_loader:

            input_ids = subpart["input_ids"].to(device) 
            attention_mask = subpart["attention_mask"].to(device)
            labels = subpart["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, predicted = torch.max(outputs, dim=1)
            # calculates loss for current batch of predictions given labels
            loss = loss_func(outputs, labels)
            # check how many of the predictions where correct
            correct_predictions += torch.sum(predicted == labels)
            # current batch loss
            losses.append(loss.item())
    
    mean_loss = np.mean(losses)
    accuracy = correct_predictions.double() / num_datapoints
    return accuracy, mean_loss


def create_predictions_BERT(model, data_loader, device):
    """
    Explanation:          This function will run model on dataloader
    
    INPUT: 
     - model:             current state of the model
     - date_loader:       the data_loader that will be used, contains both samples and labels
     - device:            device for gpu cuda

    OUTPUT:
    - accuracy:           accuracy of model 
    - mean_loss:          mean loss of model 
    """
    model = model.eval()
    pred_labels = np.array([])
    # turn off gradients calculation so we don't update model while running test data through model
    with torch.no_grad():
        for subpart in data_loader:
            input_ids = subpart["input_ids"].to(device) 
            attention_mask = subpart["attention_mask"].to(device)
            
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, predicted = torch.max(outputs, dim=1)
            # adds label to list of labels
            predicted = predicted.cpu().detach().numpy()
            pred_labels = np.append(pred_labels, predicted, axis=0)
            
            
    return pred_labels


def create_data_loader_BERT(data_list, labels, tokenizer, max_len, batch_size):
    """
    Explanation:    Creates a TwitterDataset_BERT and then a data loader from that dataset

    INPUT: 
     - data_list:   list of strings, one sentence per index
     - labels:      labels to the corresponding sentences
     - tokenizer:   tokenizer used to tokenize dataset
     - max_len:     maximum number of tokens for a given sentence
     - batch_size:  batch size for data loader and dataset

    OUTPUT:
     - data_loader: outputs a dataloader for TwitterDataset_BERT
    """
    dataset = TwitterDataset_BERT(
        samples = data_list,
        labels = labels,
        max_len = max_len,
        tokenizer = tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    return data_loader


def create_data_loader_MiniLM(pos_data_embed, neg_data_embed, batch_size, test_size):
    """
    EXPLANATION:        This function will take in positive and negative sentences and return data loaders for all data and lite version of data

    INPUT:
     - pos_data_embed:  list of sentence embeddings for positive labeled sentences
     - neg_data_embed:  list of sentence embeddings for negative labeled sentences
     - batch_size:      batch size for the data loader
     - test_size:       how big proportion of the data should be test data

    OUTPUT:
     - train_loader:        train data loader for full dataset
     - train_loader_lite:   train data loader for subpart of dataset
     - test_loader:         test data loader for full dataset
     - test_loader_lite:    test data loader for subpart of dataset

    """

    # appending correct labels to the corresponding vectors
    pos_data_embed_list = [(vector, 1) for vector in pos_data_embed]
    neg_data_embed_list = [(vector, 0) for vector in neg_data_embed]

    # creating one list we can shuffle and split into train and test
    full_data_embed_list = pos_data_embed_list + neg_data_embed_list
    train_data, test_data = split_dataset(full_data_embed_list, 1-test_size)

    # collecting samples and labels
    train_samples = np.array([x[0] for x in train_data])
    train_labels = np.array([x[1] for x in train_data])
    train_samples_tensor  = torch.tensor(train_samples)
    train_labels_tensor = torch.tensor(train_labels)

    test_samples = np.array([x[0] for x in test_data])
    test_labels = np.array([x[1] for x in test_data])
    test_samples_tensor = torch.tensor(test_samples)
    test_labels_tensor = torch.tensor(test_labels)

    train_data_labels = train_labels_tensor.type(torch.LongTensor) # can't be float
    test_data_labels = test_labels_tensor.type(torch.LongTensor) # can't be float

    # lite data: creating the DataLoaders for smaller training and smaller test set
    train_samples_tensor_lite = train_samples_tensor[:int(len(train_samples_tensor)*0.05)]
    test_samples_tensor_lite = test_samples_tensor[:int(len(test_samples_tensor)*0.05)]
    train_labels_lite = train_data_labels[:len(train_samples_tensor_lite)]
    test_labels_lite = test_data_labels[:len(test_samples_tensor_lite)]

    train_dataset_lite = TensorDataset(train_samples_tensor_lite, train_labels_lite)
    train_loader_lite = DataLoader(dataset=train_dataset_lite, batch_size=batch_size,  shuffle=True)

    test_dataset_lite = TensorDataset(test_samples_tensor_lite, test_labels_lite)
    test_loader_lite = DataLoader(dataset=test_dataset_lite, batch_size=batch_size,  shuffle=False)

    # full data: creating the DataLoaders for full training and full test data
    train_dataset = TensorDataset(train_samples_tensor, train_data_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,  shuffle=True)

    test_dataset = TensorDataset(test_samples_tensor, test_data_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, train_loader_lite, test_loader, test_loader_lite, train_data_labels, train_labels_lite, test_data_labels, test_labels_lite


def create_scaled_matrix_tensor(
    train_data_full,
    train_data_lite,
    test_data_full,
    test_data_lite,
    word_embed_full,
    word_embed_lite,
    vec_dim,
    p_norm,
    dim_norm):

    """
    EXPLANATION:                Will turn full and lite dataset (train and test) into scaled tensors

    INPUT:
     - train_data_full:         full training data
     - train_data_lite:         subpart of full training data
     - test_data_full:          full test data
     - test_data_lite:          subpart of full test data
     - word_embed_full:         word embeddings dictionary containing all words in corpus
     - word_embed_lite:         word embeddings dictionary containing all word in subpart of corpus
     - vec_dim:                 dimensions of word embedding vector
     - p_norm:                  p value for normalize function
     - dim_norm:                dim value for normalize function
     
    OUTPUT:
     - s__train_full: scaled matrix for training of full dataset
     - s__train_lite: scaled matrix for training of subpart of full dataset
     - s__test_full:  scaled matrix for testing of full dataset
     - s__test_lite:  scaled matrix for testing of subpart of full dataset
    """
    # full version of dataset used for training the model
    matrix_train_full, labels_train_full = word_vec_to_aggregated_word_embeddings(
        train_data_full, 
        word_embed_full,
        vec_dim)
    matrix_train_full_tensor = torch.tensor(matrix_train_full).type(torch.float)
    s_train_full = normalize(matrix_train_full_tensor, p=p_norm, dim=dim_norm)

    matrix_test_full, labels_test_full = word_vec_to_aggregated_word_embeddings(
        test_data_full,
        word_embed_full,
        vec_dim)
    matrix_test_full_tensor = torch.tensor(matrix_test_full).type(torch.float)
    s_test_full = normalize(matrix_test_full_tensor, p=p_norm, dim=dim_norm)

    # small version of dataset used for param optimizing
    matrix_train_lite, labels_train_lite = word_vec_to_aggregated_word_embeddings(
        train_data_lite,
        word_embed_lite,
        vec_dim)
    matrix_train_lite_tensor = torch.tensor(matrix_train_lite).type(torch.float)
    s_train_lite = normalize(matrix_train_lite_tensor, p=p_norm, dim=dim_norm)

    matrix_test_lite, labels_test_lite = word_vec_to_aggregated_word_embeddings(
        test_data_lite,
        word_embed_lite,
        vec_dim)
    matrix_test_lite_tensor = torch.tensor(matrix_test_lite).type(torch.float)
    s_test_lite = normalize(matrix_test_lite_tensor, p=p_norm, dim=dim_norm)

    return (s_train_full,
        s_train_lite,
        s_test_full,
        s_test_lite,
        labels_train_full,
        labels_train_lite,
        labels_test_full, 
        labels_test_lite)

def create_data_loader_GloVe(
    train,
    test, 
    labels_train,
    labels_test,
    batch_size):
    
    
    labels_train_full = labels_train.type(torch.LongTensor) # can't be float
    labels_test_full = labels_test.type(torch.LongTensor) # can't be float

    # already shuffled
    train_dataset = TensorDataset(train, labels_train_full)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,  shuffle=False)
    test_dataset = TensorDataset(test, labels_test_full)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,  shuffle=False)

    return train_loader, test_loader

    

def split_dataset(dataset, train_percent):
    """
    Explanation:         Shuffles and splits dataset into training part and testing part
    
    INPUT:           
     - dataset           Dataset of type TwitterDataset
     - train_percent     % of full dataset that will be used for training, rest for testing
     
    OUPUT:
     - train_data:       Shuffled list of (vector, label)
     - test_data:        Shuffled list of (vector, label)
    """
    # Gets index which we will split on
    train_last_index = int(len(dataset)*train_percent)
    
    # each elements in list shuffled_data contains (vector, label)
    random.seed(RANDOM_SEED)
    shuffled_data = list(dataset)
    shuffle(shuffled_data) # shuffle in place
    
    train_data = shuffled_data[:train_last_index]
    test_data = shuffled_data[train_last_index:]
    print("Number of elements in train_data is: ", len(train_data))
    print("Number of elements in test_data is: ", len(test_data))
    
    return train_data, test_data


def word_vec_to_word_embeddings(dataset, wordembeddings_dct, vec_size, dimension):
    """
    Explanation:             Translates dataset into flattened wordembeddings

    INPUT:  
     - dataset:              list of sentences and their corresponding label
     - wordembeddings_dct:   dictionary containing translations from word-index to word-embedding
     - vec_sice:             longest vector in the dataset
     - dimension:            dimensions of the word_embeddings

    OUTPUT:
     - matrix:               2D matrix with (dimension * longest sentences) number of features
     - labels:               labels for each of the sentences
    """
    matrix = torch.zeros(len(dataset), vec_size*dimension)
    labels = torch.zeros(len(dataset))

    for sentence_index, (sentence, label) in tqdm(enumerate(dataset)):
        
        #need to be same length as dim from word embedding
        current_sentence = torch.zeros(vec_size,dimension)
        
        for word_index, word in enumerate(sentence):
            current_sentence[word_index] = wordembeddings_dct[word]
        matrix[sentence_index] = torch.flatten(current_sentence)
        labels[sentence_index] = label
    return matrix, labels


def word_vec_to_aggregated_word_embeddings(dataset, wordembeddings_dct, dimension):
    """
    Explanation:             Translates dataset into an aggregated sum of all the word-embeddings for each setnence

    INPUT:  
     - dataset:              list of sentences and their corresponding label
     - wordembeddings_dct:   dictionary containing translations from word-index to word-embedding
     - dimension:            dimensions of the word_embeddings

    OUTPUT:
     - matrix:               2D matrix with dimension number of features
     - labels:               labels for each of the sentences
    """
    matrix = torch.zeros(len(dataset), dimension)
    labels = torch.zeros(len(dataset))
    
    for sentence_index, (sentence, label) in tqdm(enumerate(dataset)):
        
        #need to be same length as dim from word embedding
        current_sentence = torch.zeros(dimension).type(torch.float)
        
        for word_index, word in enumerate(sentence):
            current_sentence += wordembeddings_dct[word].type(torch.float)
        
        # Dividing by sentence length so we long sentences give more weight just because their long
        matrix[sentence_index] = torch.flatten(current_sentence)/len(sentence) 
        labels[sentence_index] = label
    return matrix, labels


def word_embeddings_extraction(data, input_labels, wordembeddings_dct, vec_size, dimension):
    """
    Explanation:             Translates dataset into flattened wordembeddings

    INPUT:  
     - data:                 list of sentences 
     - input_labels:         list of labels
     - wordembeddings_dct:   dictionary containing translations from word-index to word-embedding
     - vec_sice:             longest vector in the dataset
     - dimension:            dimensions of the word_embeddings

    OUTPUT:
     - matrix:               2D matrix with (dimension * longest sentences) number of features
     - labels:               labels for each of the sentences
    """
    assert len(data) == len(input_labels)

    matrix = torch.zeros(len(data), vec_size*dimension)
    labels = torch.zeros(len(input_labels))
    
    for sentence_index, (sentence, label) in enumerate(zip(data,input_labels)):
        
        #need to be same length as dim from word embedding
        current_sentence = torch.zeros(vec_size,dimension)
        
        for word_index, word in enumerate(sentence):
            current_sentence[word_index] = wordembeddings_dct[word]
        matrix[sentence_index] = torch.flatten(current_sentence)
        labels[sentence_index] = label

    return matrix, labels


def current_batch_train(lr_model, dataset, word_embeddings, vec_size, dimensions, batch_size, index):
    """
    Explanation:             Trains one batch of data with partial_fit and returns the model

    INPUT:
     - lr_model:              linear model that will be used for current training batch
     - dataset:               dataset containing list of indexes corresponding to specific words
     - wordembeddings:        dictionary containing translations from word-index to word-embedding
     - vec_sice:              longest vector in the dataset
     - dimension:             dimensions of the word_embeddings
     - batch_size:            number of sentences that will be changed into long word-embeddings
     - index:                 current iteration the batch training is currently on

    OUTPUT:
     - lr_model:              linear model that has been trained on current batch
    """
    data_xy_subset = dataset[(batch_size)*index:(batch_size)*(index+1)]
    
    data_subset = [tuple_[0] for tuple_ in data_xy_subset]
    label_subset = [tuple_[1] for tuple_ in data_xy_subset]
    
    X, y = word_embeddings_extraction(
        data_subset,
        label_subset,
        word_embeddings,
        vec_size,
        dimensions)
        
    lr_model.partial_fit(X, y, classes=np.unique(y))
    
    return lr_model


def batch_sized_linear_model(batch_size, dataset, word_embeddings, vec_size, dimensions, alpha):
    """
    Explanation:             Trains a linear model in batch-size chunks

    INPUT:
     - batch_size:            number of sentences that will be changed into long word-embeddings
     - dataset:               dataset containing list of indexes corresponding to specific words
     - wordembeddings:        dictionary containing translations from word-index to word-embedding
     - vec_sice:              longest vector in the dataset
     - dimension:             dimensions of the word_embeddings

    OUTPUT:
     - lr_model:              linear model that has been trained on all the batches
    """
    iterations = ceil(len(dataset)/batch_size)
    lr_model = SGDClassifier(loss="log", n_jobs=-1, random_state=RANDOM_SEED, warm_start=True, penalty='l2', alpha=alpha)
    
    for i in range(iterations):
        lr_model = current_batch_train(lr_model, dataset, word_embeddings, vec_size, dimensions, batch_size, i)       
        
    return lr_model


def output_numpy_array_from_model_training(loader, model):
    """
    Explanation:              Will create prediction from a given model and a data loader.

    INPUT:
     - loader:                data loader with dataset that will be used
     - model:                 neural network model that will be used
     

    OUTPUT:
     - output_pred_nn:        numpy array with output from model
    """
    output_pred_nn = np.array([])
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            _, pred = output.max(1)
            pred = pred.numpy()
            output_pred_nn = np.concatenate((output_pred_nn,pred))   

    return output_pred_nn


def output_numpy_array_from_model_submission(loader, model):
    """
    Explanation:              Will create prediction specifically for submission from a given model and a data loader.

    INPUT:
     - loader:                data loader with dataset that will be used
     - model:                 neural network model that will be used
     

    OUTPUT:
     - output_pred_nn:        numpy array with output from model
    """
    output_pred_nn = np.array([])
    with torch.no_grad():
        for x in loader:
            output = model(x)
            _, pred = output.max(1)
            pred = pred.numpy()
            output_pred_nn = np.concatenate((output_pred_nn,pred))   

    return output_pred_nn


def get_count_of_longest_sentence(dataset):
    """
    Explanation:              Will iterate over the whole dataset and find the length of the longest sentence

    INPUT:
     - dataset:               dataset of type TwitterDataset
     

    OUTPUT:
     - max_len:               length of the longest sentence, (int)
    """
    max_len = 0
    for vec, label in dataset:
        if len(vec) > max_len:
            max_len = len(vec)


    test_dataset_iter = iter(dataset.submission_dataset)
    for vec, id_  in test_dataset_iter:
        if len(vec) > max_len:
            max_len = len(vec)
    return max_len

