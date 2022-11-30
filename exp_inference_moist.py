# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:56:55 2020

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
    if y[i]>15.5:
        TGA_data.append(x[i])
    if y[i]>60:
        break
    
TGA_data_new=[]    
for i in range(50):
    TGA_data_new.append(TGA_data[i*89]*mo/100)    
        
        

#Set random seed for reproducibility 
np.random.seed(1)



# Set the inital guess.
A1=math.log10(3.17e6)
A2=math.log10(2.13e19)
Am=math.log10(1.17e7)
E1=math.log10(8.616e4)
E2=math.log10(2.61e5)
Em=math.log10(5.12e4)
alpha=10


sigma=0.1
   
A1=[4]
A2=[19.6]
E1=[4.75]
E2=[5.4]
sigma=[sigma]
Am=[Am]
Em=[Em]

   
    
sol=np.array(odt.TGAexpmoist(math.pow(10,A1[0]), math.pow(10,A2[0]),math.pow(10,Am[0]), math.pow(10,E1[0]), math.pow(10,E2[0]), math.pow(10,Em[0]),alpha))

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
    Amt=scist.norm.rvs(loc=Am[-1],scale=0.05,size=1)[0]
    Emt=scist.norm.rvs(loc=Em[-1],scale=0.005,size=1)[0]
  
    sol=odt.TGAexpmoist(math.pow(10,A1t), math.pow(10,A2t),math.pow(10,Amt), pow(10,E1t), pow(10,E2t),pow(10,Emt), alpha)
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
            Am.append(Amt)
            Em.append(Emt)
            sigma.append(sigmat)
            p1=p2
            acceptance.append(len(A1)/(i+2))
            ps.append(p1)
            
with open('output/data_expmoist.csv', 'w') as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerows([A1,A2,E1,E2,Am,Em,sigma])
