#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv 
import numpy as np
import math
import matplotlib.pyplot as plt
import random


# In[2]:


def read_data(file):
    
    feature_vecs = []
    labels = []
    
    with open(file,"r") as csvfile:
        
        csvreader = csv.reader(csvfile)
        
        for row in csvreader:
            feature_vecs.append(row[0:len(row)-1])
            labels.append(row[len(row)-1])
            
    return (feature_vecs,labels)
            
            


# In[3]:


class Layer_info: #For hidden layers and output layer
    
    def __init__(self):
        self.units = None
        self.incoming_weights = None #All incoming weights
        self.bias = None #bias terms
        self.net_values = None #weighted summation + bias
        self.activation_output = None #applying avtivation function on net_value
        self.deltas = None #useful in error backpropagation
        self.deriv_weights = None #used to store partial derivatives of weights
        self.deriv_bias = None  #used to store partial derivatives of bias
        self.update_for_weights = None 
        self.update_for_bias = None

class FeedForwardNeuralNetwork:
    
    def __init__(self,data_file,hidden_layer_lst):
        self.layers_lst = None
        self.train_patterns = None
        self.train_labels = None
        self.test_patterns = None
        self.test_labels = None
        self.initialize(data_file,hidden_layer_lst)
        self.learn_parameters_plain_grad_descent()
        
    def initialize(self,data_file,hidden_layer_lst):
       
        
        
        data = read_data(data_file)
        data_features_lst = data[0]
        data_labels_lst = data[1]
        train_features_mat = np.matrix(data_features_lst[0:5600])
        self.train_patterns = train_features_mat.astype(np.float)
        self.train_labels = [int(i) for i in data_labels_lst[0:5600]]
        test_features_mat = np.matrix(data_features_lst[5600:])
        self.test_patterns = test_features_mat.astype(np.float)
        self.test_labels = [int(i) for i in data_labels_lst[5600:]] #If some words etc. , take care to convert to 0-9.
        
        
        mean_mat = self.train_patterns.mean(0)
        std_mat = self.train_patterns.std(0)
        for j in range(self.train_patterns.shape[1]):
            for i in range(self.train_patterns.shape[0]):
                if std_mat[0,j] == 0 and mean_mat[0,j] == 0 :
                    pass
                else:
                    self.train_patterns[i,j] = (self.train_patterns[i,j] - mean_mat[0,j])/std_mat[0,j]
                    
        for j in range(self.test_patterns.shape[1]):
            for i in range(self.test_patterns.shape[0]):
                if std_mat[0,j] == 0 and mean_mat[0,j] == 0 :
                    pass
                else:
                    self.test_patterns[i,j] = (self.test_patterns[i,j] - mean_mat[0,j])/std_mat[0,j]
        
        K = len(set(self.train_labels)) #total number of classes
        d = train_features_mat[0,:].shape[1] #number of features in a pattern
        
        self.layers_lst = []
        #fill layers_lst
        
        for i in range(len(hidden_layer_lst) + 1):
            layer = Layer_info()
            if(i < len(hidden_layer_lst)):
                layer.units = hidden_layer_lst[i]
            else:
                layer.units = K
            rows = 0
            cols = layer.units
            if(i == 0):
                rows = d
            else:
                rows = hidden_layer_lst[i - 1]
            layer.incoming_weights = 0.00001*np.random.randn(rows,cols)
            layer.bias = 0.00001*np.random.randn(layer.units,1)
            layer.net_values = np.zeros( (layer.units,1) )
            layer.deltas = np.zeros( (layer.units,1) )
            layer.activation_output = np.zeros( (layer.units,1) )
            layer.deriv_weights = np.zeros((rows,cols))
            layer.deriv_bias = np.zeros((layer.units,1))
            layer.update_for_weights = np.zeros((rows,cols))
            layer.update_for_bias = np.zeros((layer.units,1))
            self.layers_lst.append(layer)
            
            
    def user_input(self):
        pass

    def classify(self,input_pattern): # X is a column vector(a pattern)
        self.forward_pass(input_pattern)
        output_vector = (self.layers_lst[len(self.layers_lst) - 1]).activation_output  #The output layer
        #print(output_vector)
        max_val = 0
        max_ind = 0
        for i in range(output_vector.shape[0]):
            if(max_val < output_vector[i,0]):
                max_val = output_vector[i,0]
                max_ind = i
                #print(i)
        return max_ind
    
    def accuracy_test(self):
        correct = 0
        for i in range(self.test_patterns.shape[0]):
            output_label = self.classify(self.test_patterns[i,:].transpose())
            test_label = self.test_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",test_label)
            if test_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                print("WRONG")
        acc = 100*correct/self.test_patterns.shape[0]
        return acc


    def accuracy_train(self):
        correct = 0
        for i in range(self.train_patterns.shape[0]):
            output_label = self.classify(self.train_patterns[i,:].transpose())
            train_label = self.train_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",train_label)
            if train_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                print("WRONG")
        acc = 100*correct/self.train_patterns.shape[0]
        return acc


    def sigmoid_vec(self,col_vector): #I/P is a column vector
        o_vec = np.matrix(col_vector)
        for i in range(col_vector.shape[0]):
            try:
                o_vec[i,0] = 1/(1 + math.exp(-1 *col_vector[i,0]))
            except OverflowError:
                o_vec[i,0] = 0.000000001
        return o_vec
    
    def deriv_sigmoid_vec(self,col_vector):
        Sigmoid_vec = self.sigmoid_vec(col_vector)
        return np.multiply(Sigmoid_vec , (1 - Sigmoid_vec))
    
    
    
    def softmax(self,col_vector):
        ans = np.matrix(col_vector)
        for i in range(col_vector.shape[0]):
            temp = np.matrix(col_vector)
            temp.fill(col_vector[i,0])
            #print(col_vector - temp)
           # ans[i,0] = 1/np.sum(np.exp(col_vector - temp))
            Sum = 0
            try:
                for j in range(col_vector.shape[0]):
                    Sum = Sum + math.exp(col_vector[j,0] - temp[j,0])
                    
                ans[i,0] = 1/Sum
            except OverflowError:
                    ans[i,0] = 0.0000000001
           
        return ans
    
    def forward_pass(self,input_pattern): #updates net_values, activation_outputs of the layers of the network, input_pattern is a column vector
        for i in range(len(self.layers_lst)):
            
            if(i == 0):
                X = input_pattern
            else:
                X = self.layers_lst[i-1].activation_output
                
            W = self.layers_lst[i].incoming_weights
            b = self.layers_lst[i].bias
            
            self.layers_lst[i].net_values = np.matmul(W.transpose(),X) + b
            
            if(i == len(self.layers_lst)-1):
                self.layers_lst[i].activation_output = self.softmax(self.layers_lst[i].net_values)
            else:
                self.layers_lst[i].activation_output = self.sigmoid_vec(self.layers_lst[i].net_values)
                
            #print("l = ",i, "  Net values:", self.layers_lst[i].net_values.transpose())
            #print("Activn output: ", self.layers_lst[i].activation_output.transpose())
            
            
    def Target(self,i): #for ith training example
        num_classes = len(set(self.train_labels))
        target = np.zeros((num_classes,1))
        
        class_i = self.train_labels[i]
        target[class_i,0] = 1
        return target

    
    
    
    def CrossEntropyLoss(self):
        Cost = 0
        for i in range(self.train_patterns.shape[0]):
            Xi = self.train_patterns[i,:].transpose()
            self.forward_pass(Xi)
            output = self.layers_lst[len(self.layers_lst) - 1].activation_output
            target = self.Target(i)
  
            Cost = Cost +  -1*np.sum(np.multiply(target,np.log(output)))
          
        
        return Cost
        
    
    def backward_pass(self,i,input_pattern):# updates deltas, deriv_weights, deriv_bias of all layers
        l = len(self.layers_lst) -1
        while(l >= 0):
            
            if(l == len(self.layers_lst)-1): #that is dealing with the output layer
                target = self.Target(i)
                output = self.layers_lst[l].activation_output
                
             
                self.layers_lst[l].deltas = target - output

                self.layers_lst[l].deriv_weights = np.matmul(self.layers_lst[l-1].activation_output , (-1 * self.layers_lst[l].deltas).transpose())

                
            else: #that is dealing with the hidden layer
               
                self.layers_lst[l].deltas = np.multiply(np.matmul(self.layers_lst[l+1].incoming_weights , self.layers_lst[l+1].deltas), self.deriv_sigmoid_vec(self.layers_lst[l].net_values))
                if(l == 0):
                    
                    self.layers_lst[l].deriv_weights = np.matmul(input_pattern , (-1 * self.layers_lst[l].deltas).transpose())
                
                else:
                 
                    self.layers_lst[l].deriv_weights = np.matmul(self.layers_lst[l-1].activation_output , (-1 * self.layers_lst[l].deltas).transpose())

            
            self.layers_lst[l].deriv_bias = -1*self.layers_lst[l].deltas
            l = l-1
            
            
    def backpropagation(self,i,input_pattern): #input_pattern is a column vector
        self.forward_pass(input_pattern)
        self.backward_pass(i,input_pattern)
        
        
            
    def learn_parameters_plain_grad_descent(self,batchsize = 1, eta = 0.01,epochs_limit = 20):
        cost_lst = []
        cost = self.CrossEntropyLoss()
        print("Starting Loss fn value:",cost)
        #cost_lst.append(cost) 
        
        bs = batchsize
        num_minibatches = math.ceil(self.train_patterns.shape[0]/bs)
        
        count_epochs = 0
        lambd = 0.9
        W_previous_update = []
        b_previous_update = []  
        while(count_epochs < epochs_limit):
            
            for j in range(num_minibatches):
                
                start = j*bs
                stop = (j + 1)*bs
                if(stop > self.train_patterns.shape[0]/bs):
                    stop = self.train_patterns.shape[0]
                
                # Now considering a mini-batch i.e. start to stop-1,update parameters
                
                for l in range(len(self.layers_lst)): #initilaize with zeroes
                    self.layers_lst[l].update_for_weights.fill(0) 
                    self.layers_lst[l].update_for_bias.fill(0)
            
                
                for i in range(start,stop):
                    Xi = self.train_patterns[i,:].transpose() # column vector(pattern)
                    self.backpropagation(i,Xi)
                
                    for l in range(len(self.layers_lst)):
                        self.layers_lst[l].update_for_weights = self.layers_lst[l].update_for_weights + self.layers_lst[l].deriv_weights
                        self.layers_lst[l].update_for_bias = self.layers_lst[l].update_for_bias + self.layers_lst[l].deriv_bias
                
                 
                for l in range(len(self.layers_lst)):
                    W_old_l = self.layers_lst[l].incoming_weights 
                    b_old_l = self.layers_lst[l].bias 
                    if(count_epochs == 0 and j == 0):
                        self.layers_lst[l].incoming_weights = self.layers_lst[l].incoming_weights + -eta*self.layers_lst[l].update_for_weights 
                        self.layers_lst[l].bias = self.layers_lst[l].bias + -eta*self.layers_lst[l].update_for_bias
                        W_previous_update.append(self.layers_lst[l].incoming_weights - W_old_l) 
                        b_previous_update.append(self.layers_lst[l].bias  - b_old_l)
                    else:
                        self.layers_lst[l].incoming_weights = self.layers_lst[l].incoming_weights + -eta*self.layers_lst[l].update_for_weights + lambd*W_previous_update[l]
                        self.layers_lst[l].bias = self.layers_lst[l].bias + -eta*self.layers_lst[l].update_for_bias + lambd*b_previous_update[l]
                        W_previous_update[l] = self.layers_lst[l].incoming_weights - W_old_l
                        b_previous_update[l] = self.layers_lst[l].bias  - b_old_l

                
               
                #cost = self.CrossEntropyLoss()
                cost = "later"
                #cost_lst.append(cost)
                
                print("After ", count_epochs*num_minibatches + j + 1 ," updation, the cost fn value is ", cost)
                
            count_epochs += 1
            
        #plt.plot(cost_lst)
        #plt.show()
                
                    
                
                
    


# In[4]:


ffn = FeedForwardNeuralNetwork("mnist_data.csv",[75])


print("\n\naccuracy on train_set is: ",ffn.accuracy_train())


# In[5]:


print("\n\nAccuracy on test_set is: ", ffn.accuracy_test())


# In[ ]:





# In[ ]:




