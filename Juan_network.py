# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:01:46 2018

@author: Erik
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#%%

class Network:
    def __init__(self, num_nodes=3, prob=0.5, eps=0.5, init_money=10, fixed_fee=1, fee_fraction=0.1, retire_thresh=0):
        self.N = num_nodes
        self.x = np.zeros(self.N) # x^t_n = 
        self.y = np.full(self.N, 1) # y^t_n = 
        self.A = np.zeros((self.N, self.N)) # A^t_{n,m} = 
        self.B = np.zeros((self.N, self.N)) # B_{n,m} = 
        self.M = np.full(self.N, init_money) 
        self.r = np.zeros(self.N) # r^t_n = 
        self.l = np.zeros(self.N) # l^t_n = 
        self.p = prob # Erdos-Renyi probability
        self.t = 0 # time
        self.eps = eps # eps = probability of recruitment
        self.pay = np.zeros(self.N) # what each node must pay
        self.fixed_fee = fixed_fee
        self.fee_fraction = fee_fraction
        self.retire_thresh = retire_thresh
        
    def erdos_renyi(self):
        """
        Constructs adjacency matrix for Erdos-Renyi random graph
        """
        # Draw N*(N-1)/2 Bernoulli RVs to fill upper triangular part of adjacency matrix
        tmp = np.random.binomial(n=1, p=self.p, size=int(self.N*(self.N-1)/2))
        k = 0
        for row in range(0,self.N-1):
            for colm in range(row+1, self.N):
                self.B[row, colm] = tmp[k]
                k += 1
        
        # Make matrix symmetric
        self.B = self.B + self.B.T - np.diag(self.B.diagonal())
    
    
    def user_adjacency(self):
        """
        Create adjacency matrices by hand for testing
        """
        self.B = np.array([[0,0,1], [0,0,1], [1,1,0]])
    
    def user_init(self):
        """
        Initialization step by hand for testing
        """
        parent = 0
        self.x[parent] = 1
        self.l[parent] = 1
        self.r[parent] = 1
        
    def initialize(self):
        """
        Randomly select which node starts the pyramid scheme
        """
        parent = np.random.randint(self.N)
        self.x[parent] = 1
        self.l[parent] = 1
        self.r[parent] = 1
        
        

    def recruitment(self):
        for node in np.random.permutation(np.argwhere(net.r==1).flatten()):
            neighbors = np.random.permutation(np.argwhere((self.B[node,] * (1-self.x) * self.y)==1).flatten())
            for neighbor in neighbors:
                if np.random.binomial(n=1, p=self.eps)==1:
                    self.x[neighbor] = 1
                    self.A[node,neighbor] = 1
                    self.l[neighbor] = self.l[node] + 1
                    self.r[neighbor] = 1
            self.r[node] = 0
            
    def money_update(self):
        self.pay = np.zeros(self.N)
        max_level = int(np.max(self.l))
        
        for level in range(max_level, 0, -1):

            indices = np.argwhere(self.l==level)
            if level==max_level:
                self.pay[indices] = self.fixed_fee * self.x[indices] * self.y[indices]
                
                tmp_pay = np.zeros(self.N)
                tmp_pay[indices] = self.pay[indices]
            elif level==1:
                self.pay[indices] = 0
            else:
                self.pay[indices] = (self.fixed_fee + self.fee_fraction*(self.A[indices,] @ tmp_pay)) * self.x[indices] * self.y[indices]
        
                tmp_pay = np.zeros(self.N)
                tmp_pay[indices] = self.pay[indices]
        
        # Make sure that the amount a node pays is not more than the amount of money that node has
        for node in range(0, len(self.pay)):
            if self.pay[node]>self.M[node]:
                self.pay[node] = self.M[node]
        
        self.M = self.M - self.pay + self.A @ self.pay

        
    def retirement_from_pyramid(self):
        self.y[np.argwhere(self.M<=self.retire_thresh)] = 0
        
        
        
        

#%% Testing
net = Network(num_nodes=3, eps=1, init_money=5, fixed_fee=1, fee_fraction=0.1, retire_thresh=0)
#net.initialize()
#net.erdos_renyi()

net.user_init()
net.user_adjacency()

#%%n
#print(net.B, net.r, net.x, net.y)
for node in np.random.permutation(np.argwhere(net.r==1).flatten()):
    print(node)
    neighbors = np.random.permutation(np.argwhere((net.B[node,] * (1-net.x) * net.y)==1).flatten())
    print(neighbors)
    for neighbor in neighbors:
        print(neighbor)
        if np.random.binomial(n=1, p=0)==1:
            net.x[neighbor] = 1
            net.A[node,neighbor] = 1
            net.l[neighbor] = net.l[node] + 1
            net.r[neighbor] = 1
    net.r[node] = 0
    
#%%
x = np.array([1,2,3,2,3])
indices = np.argwhere(x>2)

A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
(A[[1,2],] @ np.array([1,1,1,1]))*np.array([1,1])*np.array([0,1])

#%% Testing
net = Network(num_nodes=200, prob=0.1, eps=0.1, init_money=5, fixed_fee=0.5, fee_fraction=0.1, retire_thresh=0)
net.initialize()
net.erdos_renyi()

#net.user_init()
#net.user_adjacency()

#%%
k = 1
while (k==1 or np.any(net.r)):
    print(k)
    net.recruitment()
    net.money_update()
    net.retirement_from_pyramid()
    k += 1