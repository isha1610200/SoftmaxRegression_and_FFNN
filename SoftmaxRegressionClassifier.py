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


class SoftmaxRegressionClassifier:
    
    def __init__(self,data_file):
        self.theta_matrix = None
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.initialize(data_file)
        #self.learn_parameters_stochastic_gradient_descent()
        self.learn_parameters_plain_grad_descent()
        
    def initialize(self,data_file):
        
        #reading training features and labels

        data = read_data(data_file)
        data_features = data[0]
        data_labels = data[1]
        
        for i in range(len(data_features)): #augmenting feature vectors to account for the bias term
            data_features[i] = [1] + data_features[i]

        train_features_mat = np.matrix(data_features[0:5500])
        self.train_features = train_features_mat.astype(np.float)

        test_features_mat = np.matrix(data_features[5500:])
        self.test_features = test_features_mat.astype(np.float)
        
        mean_mat = self.train_features.mean(0)
        std_mat = self.train_features.std(0)
        for j in range(self.train_features.shape[1]):
            for i in range(self.train_features.shape[0]):
                if std_mat[0,j] == 0:
                    pass
                else:
                    self.train_features[i,j] = (self.train_features[i,j] - mean_mat[0,j])/std_mat[0,j]


        for j in range(self.test_features.shape[1]):
            for i in range(self.test_features.shape[0]):
                if std_mat[0,j] == 0:
                    pass
                else:
                    self.test_features[i,j] = (self.test_features[i,j] - mean_mat[0,j])/std_mat[0,j]                     
        
        self.train_labels = [int(i) for i in data_labels[0:5500]]
        
        self.test_labels = [int(i) for i in data_labels[5500:]] #If some words etc. , take care to convert to 0-9.

 
        #initialize theta matrix with zeroes
        K = len(set(self.train_labels)) #total number of classes
        n = self.train_features[0,:].shape[1] #length of the augemented feature vector
        self.theta_matrix = np.zeros([n,K],dtype = float)
        #print(n,K)
        
        
    def compute_score_vector(self,X): 
        score_vec =  np.matmul(self.theta_matrix.transpose(),X)
        return score_vec  #also a column vector
    
    def apply_softmax(self,X): #May have overflow etc.
        X = np.exp(X)
        X = X/np.sum(X)
        return X
    
    def compute_probability_vector(self,X):
        return self.apply_softmax(self.compute_score_vector(X))
    
    def classify(self,X): #X is a column vector
        prob_vec = self.compute_probability_vector(X) #find place with maximum prob.
        max_ind = 0
        max_prob = 0
        #print(prob_vec.shape)
        for i in range(prob_vec.shape[0]):
            if prob_vec[i,0] > max_prob:
                max_prob = prob_vec[i,0]
                max_ind = i
                
        return max_ind #and this is the predicted class
    
    def user_input(self):
        pass
    
    def accuracy_test(self):
        correct = 0
        for i in range(self.test_features.shape[0]):
            output_label = self.classify(self.test_features[i,:].transpose())
            test_label = self.test_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",test_label)
            if test_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                print("WRONG")
        acc = 100*correct/self.test_features.shape[0]
        return acc
    
    def accuracy_train(self):
        correct = 0
        for i in range(self.train_features.shape[0]):
            output_label = self.classify(self.train_features[i,:].transpose())
            train_label = self.train_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",train_label)
            if train_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                print("WRONG")
        acc = 100*correct/self.train_features.shape[0]
        return acc
    
    
    
    def Indicator_func(self,actual_label, Class):
        if actual_label == Class:
            return 1
        else:
            return 0
        
    def Probability(self,Xi,k): #Xi is column vector
        Nr = 1
        Dr = 0
        K = len(set(self.test_labels))
        #print("shape:",self.theta_matrix[:,k].ndim) #It is like a 1-D array
        for j in range(K):
            try:
                Dr = Dr + math.exp(np.matmul(self.theta_matrix[:,j] , Xi) - np.matmul(self.theta_matrix[:,k] , Xi))
            except OverflowError:
                return 0.0000000000001
        if(Nr/Dr == 0):
            print("Nr/Dr = ",Nr/Dr)
            return 0.00000000000001
        return Nr/Dr
        
        
    def Current_CrossEntropyLoss(self):
        cost = 0
        K = len(set(self.test_labels)) #total number of classes
        N = self.train_features.shape[0] # total number of training examples
        
        #print(self.theta_matrix)
        for i in range(N): #for each training example
            Xi = self.train_features[i,:].transpose() #A column vector
            cst = 0
            for k in range(K):
                if(self.Indicator_func(self.train_labels[i],k) == 1):
                    cst = cst + -1*self.Indicator_func(self.train_labels[i],k)*math.log(self.Probability(Xi,k))
            cost = cost + cst
        return cost        
                
        
    
    def learn_parameters_stochastic_gradient_descent(self,batchsize = 10, eta = 0.001,epsilon = 0.01,max_iter= 1600):
        cost_lst = []
        cost = self.Current_CrossEntropyLoss()
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost)
        K = len(set(self.test_labels)) #total number of classes
        n = self.train_features[0,:].shape[1] #length of the augemented feature vector
        temp_theta_matrix = np.zeros([n,K],dtype = float)  
        
        bs = batchsize
    
        lst = []
        for i in range(self.train_features.shape[0]):
            lst.append(i)
        count_iter = 0
        
        while(count_iter < max_iter):
            update_vector = []
            rand_sample = random.sample(lst,bs)
            # Now considering a stochastic mini-batch i.e. start to stop-1,update theta_matrix
            for k in range(len(set(self.train_labels))): #For this mini-batch, update the theta_matrix, basically update theta_0, theta_1, ..., thtea_K-1.
                    
                Sum = np.zeros([self.train_features.shape[1],1],dtype = float)
                    
                for i in rand_sample:
                    Xi = self.train_features[i,:].transpose()
                    Sum = Sum +  -1*Xi*(self.Indicator_func(self.train_labels[i],k) - self.Probability(Xi,k))
                        
                Grad_cost_fn = Sum 
                    
                #print(Sum.shape)
                #print(self.theta_matrix[:,k].shape)
                    
                temp_theta_matrix[:,k] = self.theta_matrix[:,k] + -eta*Grad_cost_fn.transpose()
                #print(Grad_cost_fn)
                update_vector = update_vector + Grad_cost_fn.tolist()
                    
            self.theta_matrix = temp_theta_matrix
            #print(np.matrix(update_vector).shape)
                
                #print(self.theta_matrix)
            #cost = self.Current_CrossEntropyLoss()
            cost = "left as it's computation is time consuming"
            cost_lst.append(cost)
            norm = np.linalg.norm(np.matrix(update_vector))
            count_iter += 1
            print("After ", count_iter ," updation, the cost fn value is ", cost, " and norm of update_vec is ", norm)
            
            
            if(norm < epsilon):
                break
                
            
        #plt.plot(cost_lst)
        #plt.show()
        
        
        
    def learn_parameters_plain_grad_descent(self,batchsize = 1, eta = 0.001,epochs_limit = 7):
        cost_lst = []
        cost = self.Current_CrossEntropyLoss()
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost)
        K = len(set(self.test_labels)) #total number of classes
        n = self.train_features[0,:].shape[1] #length of the augemented feature vector
        temp_theta_matrix = np.zeros([n,K],dtype = float)  
        
        bs = batchsize
        num_minibatches = math.ceil(self.train_features.shape[0]/bs)
        
        count_epochs = 0
        while(count_epochs < epochs_limit):
            
            for j in range(num_minibatches):
                
                start = j*bs
                stop = (j + 1)*bs
                if(stop > self.train_features.shape[0]/bs):
                    stop = self.train_features.shape[0]
                
                # Now considering a mini-batch i.e. start to stop-1,update theta_matrix
                
                for k in range(len(set(self.train_labels))): #For this mini-batch, update the theta_matrix, basically update theta_0, theta_1, ..., thtea_K-1.
                    
                    Sum = np.zeros([self.train_features.shape[1],1],dtype = float)
                    
                    for i in range(start,stop):
                        Xi = self.train_features[i,:].transpose()
                        Sum = Sum +  -1*Xi*(self.Indicator_func(self.train_labels[i],k) - self.Probability(Xi,k))
                        
                    Grad_cost_fn = Sum 
                    
                    #print(Sum.shape)
                    #print(self.theta_matrix[:,k].shape)
                    
                    temp_theta_matrix[:,k] = self.theta_matrix[:,k] + -eta*Grad_cost_fn.transpose()
                    #print(Grad_cost_fn)
                    
                self.theta_matrix = temp_theta_matrix
                
                #print(self.theta_matrix)
                #cost = self.Current_CrossEntropyLoss()
                cost = "left as it's computation is time consuming"
                cost_lst.append(cost)
                print("After ", count_epochs*num_minibatches + j + 1 ," updation, the cost fn value is ", cost)
                
            count_epochs += 1
            
        #plt.plot(cost_lst)
        #plt.show()
                
                    
                
                
        


# In[4]:


softmax_classifier = SoftmaxRegressionClassifier("mnist_data.csv")
print("Accuracy over training set is:",softmax_classifier.accuracy_train())
print("Accuarcy over test set is:",softmax_classifier.accuracy_test())


# In[ ]:





# In[ ]:




    
    


# In[ ]:




