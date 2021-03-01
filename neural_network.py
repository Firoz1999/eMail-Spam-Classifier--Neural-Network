# -- coding: utf-8 --
import random
import pprint
import time
import sys
from collections import Counter
import numpy as np
import pandas as pd

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={"v1":"label", "v2":"text"})
data["label_tag"] = data.label.map({'ham':0, 'spam':1})
data.label.value_counts()

training_data = data[0:4572]
training_data_length = len(training_data.label)
test_data = data[-1000:]
test_data_length = len(test_data.label)

spam_counts = Counter()
ham_counts = Counter()
total_counts = Counter()
spam_ham_ratios = Counter()

for i in range(training_data_length):
    if(training_data.label[i] == 0):
        for word in training_data.text[i].split(" "):
            ham_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in training_data.text[i].split(" "):
            spam_counts[word] += 1
            total_counts[word] += 1
            
for word,count in list(total_counts.most_common()):
    if(count > 100):
        spam_ham_ratio = spam_counts[word] / float(ham_counts[word]+1)
        spam_ham_ratios[word] = spam_ham_ratio

for word,ratio in spam_ham_ratios.most_common():
    if(ratio > 1):
        spam_ham_ratios[word] = np.log(ratio)
    else:
        spam_ham_ratios[word] = -np.log((1 / (ratio+0.01)))

#Transform Text into Numbers
vocab = set(total_counts.keys())
print(vocab)
vocab_size = len(vocab)

#setting up vocabulary vector
vocab_vector = np.zeros((1, vocab_size))

# Dictionary creation// Maps a word to its column in the vocab_vector
word_column_dict = {}
#print(vocab)
for i, word in enumerate(vocab):
    word_column_dict[word] = i

class SpamClassificationNeuralNetwork(object):
    def __init__(self, training_data, num_hidden_nodes = 10, num_epochs = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.pre_process_data(training_data)
        
        self.num_features = len(self.vocab)
        self.vocab_vector = np.zeros((1, len(self.vocab)))
        self.num_input_nodes = self.num_features
        self.num_hidden_nodes = num_hidden_nodes
        self.num_epochs = num_epochs
        self.num_output_nodes = 1
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_i_h = np.random.randn(self.num_input_nodes, self.num_hidden_nodes)
        self.weights_h_o = np.random.randn(self.num_hidden_nodes, self.num_output_nodes)
        
    
    def feedforward(self, text):
        ### Forward pass ###
        # Input Layer
        self.update_input_layer(text)
        # Hidden layer
        self.hidden_layer = self.vocab_vector.dot(self.weights_i_h)
        # Output layer
        self.output_layer = self.sigmoid(self.hidden_layer.dot(self.weights_h_o))
        return self.output_layer
        
    def backpropagate(self,label):
        ### Backward pass ###
        # Output error
        output_layer_error = self.output_layer - label 
        output_layer_delta = output_layer_error * self.sigmoid_derivative(self.output_layer)

        # Backpropagated error - to the hidden layer
        hidden_layer_error = output_layer_delta.dot(self.weights_h_o.T)
        # hidden layer gradients - no nonlinearity so it's the same as the error
        hidden_layer_delta = output_layer_error 

        # update the weights - with grdient descent
        self.weights_h_o -= self.hidden_layer.T.dot(output_layer_delta) * self.learning_rate 
        self.weights_i_h -= self.vocab_vector.T.dot(hidden_layer_delta) * self.learning_rate 
        
        if(np.abs(output_layer_error) < 0.5):
                self.correct_so_far += 1

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x * (1 - x)
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.correct_so_far = 0
            start = time.time()

            for i in range(len(training_data)):
                # Forward and Back Propagation
                self.feedforward(training_data["text"][i])
                self.backpropagate(training_data["label_tag"][i])

                samples_per_second = i / float(time.time() - start + 0.001)

                sys.stdout.write("\rEpoch: "+ str(epoch)
                                 +" Progress: " + str(100 * i/float(len(training_data)))[:4] 
                                 + " % Speed(samples/sec): " + str(samples_per_second)[0:5] 
                                 + " #Correct: " + str(self.correct_so_far) 
                                 + " #Trained: " + str(i+1) 
                                 + " Training Accuracy: " + str(self.correct_so_far * 100 / float(i+1))[:4] + "%")
            print("")
        
    def pre_process_data(self, training_data):
        vocab = set()
        for review in training_data["text"]:
            for word in review.split(" "):
                vocab.add(word)   
        self.vocab = list(vocab)
        self.word_to_column = {}
        for i, word in enumerate(self.vocab):
            self.word_to_column[word] = i
            
    def update_input_layer(self, text):
        global vocab_vector
        # clear out previous state, reset the vector to be all 0s
        self.vocab_vector *= 0
        for word in text.split(" "):
            if word in word_column_dict.keys():
                self.vocab_vector[0][word_column_dict[word]] += 1
                  
nn = SpamClassificationNeuralNetwork(training_data, num_epochs = 10, learning_rate=0.01)
nn.train()
count=0
for i in range(4572,5572):
    predicted=nn.feedforward(test_data["text"][i])
    predicted=round(predicted[0][0],5)
    if(predicted>=0.5 and test_data["label_tag"][i]==1):
        count+=1
    if(predicted<0.5 and test_data["label_tag"][i]==0):
        count+=1    
    print(test_data["label_tag"][i],"       ",predicted)
    
print(count)
