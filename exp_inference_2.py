# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:56:15 2020

Determines parameter estimates with unknown reaction order.

@author: mtb980
"""


import ODE_TGA as odt
import scipy.stats as scist
#import matplotlib.pyplot as plt
import numpy as np
import math 
import csv

x = []
y = []

mo=98.36

with open('data/TGA_data.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
            x.append(float(row[3]))
            y.append(float(row[1]))

TGA_data=[]  

dat=np.loadtxt('data/TGA_data.txt',delimiter=';')      
for i in range(len(y)):
    if y[i]>25:
        TGA_data.append(x[i])
    if y[i]>60:
        break
    
TGA_data_new=[]    
for i in range(50):
    TGA_data_new.append(TGA_data[i*70]*mo/100)    
        
        

#Set random seed for reproducibility 
np.random.seed(1)



# Set the inital guess.
A1=math.log10(3.17e6)
A2=math.log10(2.13e19)
E1=math.log10(8.616e4)
E2=math.log10(2.61e5)
alpha=10
n1=1
n2=1

sigma=0.1
   
A1=[A1]
A2=[A2]
E1=[E1]
E2=[E2]
sigma=[sigma]
n1=[n1]
n2=[n2]

def prior(theta)       #function that will input some prior information
    p=scist.norm(theta)
    
    
sol=np.array(odt.TGAexp2(math.pow(10,A1[0]), math.pow(10,A2[0]), math.pow(10,E1[0]), math.pow(10,E2[0]), n1[0],n2[0],alpha))

p1=scist.multivariate_normal.logpdf(TGA_data_new,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))
acceptance=[]
ps=[p1]
betas=[]
for i in range(60000):
    A1t=scist.norm.rvs(loc=A1[-1],scale=0.05,size=1)[0] #log normal variation 
    A2t=scist.norm.rvs(loc=A2[-1],scale=0.05,size=1)[0]
    E1t=scist.norm.rvs(loc=E1[-1],scale=0.005,size=1)[0]
    E2t=scist.norm.rvs(loc=E2[-1],scale=0.005,size=1)[0]
    sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.001,size=1)[0]
    n1t=scist.geom.rvs(1/2)
    n2t=scist.geom.rvs(1/2)
  
    sol=odt.TGAexp2(math.pow(10,A1t), math.pow(10,A2t), pow(10,E1t), pow(10,E2t),n1t,n2t, alpha)
    if sigmat > 0:
        p2=scist.multivariate_normal.logpdf(TGA_data_new,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigmat))
        p=p2-p1
        beta=math.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
        betas.append(beta)
        if p>beta:
            A1.append(A1t)
            A2.append(A2t)
            E1.append(E1t)
            E2.append(E2t)
            n1.append(n1t)
            n2.append(n2t)
            sigma.append(sigmat)
            p1=p2
            acceptance.append(len(A1)/(i+2))
            ps.append(p1)
            
with open('output/data_exp2.csv', 'w') as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerows([A1,A2,E1,E2,n1,n2,sigma])
