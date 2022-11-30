#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:40:46 2020

A simple TGA model with one reaction.

@author: matt
"""

import ODE_TGA as odt
import scipy.stats as scist
import matplotlib.pyplot as plt
import numpy as np
import math 
import csv

#Set random seed for reproducibility 
np.random.seed(1)

# Define the true parameters.
A=math.log10(3.17e6)
E=math.log10(8.616e4)

alpha=10

#Define the solution 
sol=np.array(odt.TGA_single(pow(10,A),pow(10,E),alpha))

#Define the true noise parameter
sigma=1

noisy_TGA=sol+np.random.normal(0,sigma,len(sol))

plt.plot(noisy_TGA)

    
A=[]
E=[]

sigma=[]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.norm.logpdf(theta[0],loc=math.log10(3.17e6))
    p2=scist.norm.logpdf(theta[1],loc=math.log10(8.61e4))
    p3=scist.norm.logpdf(math.log10(theta[2]),loc=-1,scale=1)
    p=p1+p2+p3
    return p3

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])
acceptance=[]
ps=[]
betas=[]
    
lengths=[0]
for j in range(4):
    np.random.seed(seed=10*j+2)
    E.append(scist.uniform.rvs(loc=3,scale=5,size=1)[0])
    Tm=scist.uniform.rvs(loc=600,scale=100)
    A.append(math.log10(odt.Pre_exp(Tm,10**E[-1],alpha)))
    sigma.append(1)
    sol=np.array(odt.TGA_single(math.pow(10,A[-1]),math.pow(10,E[-1]),alpha))
    p1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[-1]))+prior(A[0],E[0],sigma[0])
    ps.append(p1)
    for i in range(30000):
        At=scist.norm.rvs(loc=A[-1],scale=0.5,size=1)[0] #log normal variation 
        Et=scist.norm.rvs(loc=E[-1],scale=0.005,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.001,size=1)[0]
        
        sol=odt.TGA_single(pow(10,At), pow(10,Et),alpha)
        if sigmat > 0:
            p2=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigmat))+prior(At,Et,sigmat)
            p=p2-p1
            beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
            betas.append(beta)
            if p>beta:
                A.append(At)
                E.append(Et)
                sigma.append(sigmat)
                p1=p2
                acceptance.append((len(A)-lengths[-1])/(i+2))
                ps.append(p1)
    lengths.append(len(A))
    



with open('output/data_single.csv', 'w') as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerows([A,E,sigma])