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
        data_features = data[0]
        data_labels = data[1]

        train_features_mat = np.matrix(data_features[0:5500])
        self.train_patterns = train_features_mat.astype(np.float)

        test_features_mat = np.matrix(data_features[5500:])
        self.test_patterns = test_features_mat.astype(np.float)

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
       
        self.train_labels = [int(i) for i in data_labels[0:5500]]
        
        
       
       
        
        self.test_labels = [int(i) for i in data_labels[5500:]] #If some words etc. , take care to convert to 0-9.

        K = len(set(self.test_labels)) #total number of classes
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
            layer.incoming_weights = np.random.randn(rows,cols)
            layer.bias = np.random.randn(layer.units,1)
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
            #print("Sno: ",i,"  output label:",output_label," test_label:",test_label)
            if test_label == output_label:
                #print("CORRECT")
                correct += 1
            else:
                #print("WRONG")
                pass
        acc = 100*correct/self.test_patterns.shape[0]
        return acc


    def accuracy_train(self):
        correct = 0
        for i in range(self.train_patterns.shape[0]):
            output_label = self.classify(self.train_patterns[i,:].transpose())
            train_label = self.train_labels[i]
            #print("Sno: ",i,"  output label:",output_label," test_label:",train_label)
            if train_label == output_label:
                #print("CORRECT")
                correct += 1
            else:
                #print("WRONG")
                pass
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
        
    def forward_pass(self,input_pattern): #updates net_values, activation_outputs of the layers of the network, input_pattern is a column vector
        for i in range(len(self.layers_lst)):
            
            if(i == 0):
                X = input_pattern
            else:
                X = self.layers_lst[i-1].activation_output
                
            W = self.layers_lst[i].incoming_weights
            b = self.layers_lst[i].bias
            
            self.layers_lst[i].net_values = np.matmul(W.transpose(),X) + b
            self.layers_lst[i].activation_output = self.sigmoid_vec(self.layers_lst[i].net_values)
            #print("l = ",i, "  Net values:", self.layers_lst[i].net_values.transpose())
            #print("Activn output: ", self.layers_lst[i].activation_output.transpose())
            
    
    def Target(self,i): #for ith training example
        num_classes = len(set(self.train_labels))
        target = np.zeros((num_classes,1))
        
        class_i = self.train_labels[i]
        target[class_i,0] = 1
        return target

    
    def SumOfSquaredError_fn(self):
        Error = 0
        for i in range(self.train_patterns.shape[0]):
            Xi = self.train_patterns[i,:].transpose()
            self.forward_pass(Xi)
            output = self.layers_lst[len(self.layers_lst) - 1].activation_output
            target = self.Target(i)
            
            #for j in range(output.shape[0]):
            #    Tj = 0
            #    if(self.train_labels[i] == j)
            #       Tj = 1
            #   Error = Error + (1/2)*((Tj - output[j,0])**2)
            
            Error = Error + np.sum((1/2)* np.square(target - output))
        
        return Error
            
            
    def backward_pass(self,i,input_pattern):# updates deltas, deriv_weights, deriv_bias of all layers
        l = len(self.layers_lst) -1
        while(l >= 0):
            
            if(l == len(self.layers_lst)-1): #that is dealing with the output layer
                target = self.Target(i)
                output = self.layers_lst[l].activation_output
                
                #print("target shape: ",target.shape ," , dim:", target.ndim)
                #print("deriv_sig shape: ",self.deriv_sigmoid_vec(self.layers_lst[l].net_values).shape, " ,dim: ", self.deriv_sigmoid_vec(self.layers_lst[l].net_values).ndim)
                self.layers_lst[l].deltas = np.multiply((target - output),self.deriv_sigmoid_vec(self.layers_lst[l].net_values)) #element wise product
                #print("Activ o/p 2nd last layer:",self.layers_lst[l-1].activation_output)
                self.layers_lst[l].deriv_weights = np.matmul(self.layers_lst[l-1].activation_output , (-1 * self.layers_lst[l].deltas).transpose())
                #self.layers_lst[l].deriv_bias = -1*self.layers_lst[l].deltas
                
                
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

        
    #This one gives low accuarcy, go for the other one, and increasing max_iter will make it very slow.    
    def learn_parameters_stochastic_grad_descent(self,batchsize = 1, eta = 0.01,epsilon = 0.01,max_iter= 10000):
        cost_lst = []
        cost = self.SumOfSquaredError_fn()
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost)
       
        bs = batchsize
    
        lst = []
        for i in range(self.train_patterns.shape[0]):
            lst.append(i)
        count_iter = 0
        
        while(count_iter < max_iter):
            
            grad_cost_fn = []
            
            rand_sample = random.sample(lst,bs)
            # Now considering a stochastic mini-batch i.e. start to stop-1,update all parametrs of the network
            
            for l in range(len(self.layers_lst)): #initilaize with zeroes
                self.layers_lst[l].update_for_weights.fill(0) 
                self.layers_lst[l].update_for_bias.fill(0)
            
            for i in rand_sample:
                
                Xi = self.train_patterns[i,:].transpose() # column vector(pattern)
                self.backpropagation(i,Xi)
                #print("backprop success")
                
                for l in range(len(self.layers_lst)):
                    self.layers_lst[l].update_for_weights = self.layers_lst[l].update_for_weights + self.layers_lst[l].deriv_weights
                    self.layers_lst[l].update_for_bias = self.layers_lst[l].update_for_bias + self.layers_lst[l].deriv_bias
            
            for l in range(len(self.layers_lst)):
                self.layers_lst[l].incoming_weights = self.layers_lst[l].incoming_weights + -eta*self.layers_lst[l].update_for_weights
                self.layers_lst[l].bias = self.layers_lst[l].bias + -eta*self.layers_lst[l].update_for_bias
                
                grad_cost_fn = grad_cost_fn + self.layers_lst[l].update_for_bias.tolist()
                for k in range(self.layers_lst[l].update_for_weights.shape[1]):
                    theta_vec = self.layers_lst[l].update_for_weights[:,k] #Doubt-->in Q1, e.g. --> Prob. -acted like a 1-D array
                    #print("shape:",theta_vec.shape)
                    assert(theta_vec.shape[1] == 1)
                    grad_cost_fn = grad_cost_fn + theta_vec.tolist()
            #print(grad_cost_fn)
            
                    
            #cost = self.SumOfSquaredError_fn()
            cost = "later"
            cost_lst.append(cost)
            norm = np.linalg.norm(np.matrix(grad_cost_fn))
            count_iter += 1
            print("After ", count_iter ," updation, the cost fn value is ", cost, " and norm of update_vec is ", norm)
            
            
            if(norm < epsilon):
                break
                
            
        #plt.plot(cost_lst)
        #plt.show()
        
        
            

    def learn_parameters_plain_grad_descent(self,batchsize = 1, eta = 0.01,epochs_limit = 150):
        cost_lst = []
        #cost = self.SumOfSquaredError_fn()
        cost = "None"
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost) 
        
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

                
                
                #cost = self.SumOfSquaredError_fn()
                #cost_lst.append(cost)
                cost = "Left as it's computation is time consuming"
                #print("After ", count_epochs*num_minibatches + j + 1 ," updation, the cost fn value is ", cost)
                print("Iteration completed:  ", count_epochs*num_minibatches + j + 1 )
                
            count_epochs += 1
            
        #plt.plot(cost_lst)
        #plt.show()
                
                    


# In[4]:


ffn = FeedForwardNeuralNetwork("mnist_data.csv",[64,10,5])

print("\n\nAccuracy on test_set is: ", ffn.accuracy_test())
print("\n\naccuracy on train_set is: ",ffn.accuracy_train())

