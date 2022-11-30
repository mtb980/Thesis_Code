#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:27:09 2021
Code that loads the output from the MCMC algorithm and conducts SMC on this.
@author: matthew
"""
from joblib import Parallel, delayed

import load_data as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as scist
import scipy.special as scisp
#from collections import namedtuple

#import Figure_generator as fg

np.random.seed(1)

def log_Pre_exp(Tm,E,alpha):
    R=8.314
    A=E/R/Tm*np.log10(np.exp(1))+np.log10(E/R/Tm**2*alpha)
    return A


def dydt_single(t, x, parameters):
    dyd=[0,0]
    dyd[0]=-x[0]*parameters[0]*math.exp(-parameters[1]/(parameters[3]*x[1]))
    dyd[1]=parameters[2]
    return dyd 

def RK4(funct, *args,IC=0,t0=0,h=0.01,n=1000):
    y0=np.array(IC);
    y=np.zeros((y0.size,n+1))
    y[:,0]=y0
    ti=t0
    for i in range(n):
        k1=np.array(funct(t=ti,x=list(y[:,i]),parameters=list(args)))
        k2=np.array(funct(t=ti+h/2,x=list(y[:,i]+k1*h/2),parameters=list(args)))
        k3=np.array(funct(t=ti+h/2,x=list(y[:,i]+k2*h/2),parameters=list(args)))
        k4=np.array(funct(t=ti+h,x=list(y[:,i]+k3*h),parameters=list(args)))
        y[:,i+1]=y[:,i]+1/6*h*(k1+2*k2+2*k3+k4)
        ti=ti+h
    return y

def TGA_single(A,E,alpha):
    #Extra model parametes
    Tf=600          # Final temperature
    R=8.314         # Ideal gas constant
    parameters=[A,E,alpha,R]
    #define the initial conditions
    m1=100          #initial mass
    m0=0.85*m1      #moistureless mass
    W0=0.45*m1     #initial mass of wustite
    T0=273          #initial temperature
    
    # deifne the time scale
    t0=0            #Initial time
    h=0.001          #step size
    tf=Tf/alpha     # Final time
    n=math.ceil((tf-t0)/h)  #number of steps
    tf=t0+h*n    
 
    sol=RK4(dydt_single,A,E,alpha,R,IC=[W0,T0],t0=0,h=h,n=n)    #Calculate the solution
    
    mass=[]
    
    datapoints=60
    for i in range(datapoints):
        mass.append(sol[0,int(10000/alpha*i)])
    
    
    return mass




def processData(data_run,filtered=True,Burnin=10000):
    data_chains=ld.assemble(data_run)
    burned_data=ld.removeBurn(data_chains, Burnin=Burnin)
    if filtered:
        filtered_data=ld.filterImportant(burned_data,columnsdrop=['acceptance'])
        Results=ld.generateData(filtered_data,burnin=True)
    else:
        Results=ld.generateData(burned_data,burnin=True)
    sample=ld.sampleAsDF(Results)
    return sample



def log_Pre_exp(Tm,E,alpha):
    R=8.314
    A=E/R/Tm*np.log10(np.exp(1))+np.log10(E/R/Tm**2*alpha)
    return A


def dydt_single(t, x, parameters):
    dyd=[0,0]
    dyd[0]=-x[0]*parameters[0]*math.exp(-parameters[1]/(parameters[3]*x[1]))
    dyd[1]=parameters[2]
    return dyd 

def RK4(funct, *args,IC=0,t0=0,h=0.01,n=1000):
    y0=np.array(IC);
    y=np.zeros((y0.size,n+1))
    y[:,0]=y0
    ti=t0
    for i in range(n):
        k1=np.array(funct(t=ti,x=list(y[:,i]),parameters=list(args)))
        k2=np.array(funct(t=ti+h/2,x=list(y[:,i]+k1*h/2),parameters=list(args)))
        k3=np.array(funct(t=ti+h/2,x=list(y[:,i]+k2*h/2),parameters=list(args)))
        k4=np.array(funct(t=ti+h,x=list(y[:,i]+k3*h),parameters=list(args)))
        y[:,i+1]=y[:,i]+1/6*h*(k1+2*k2+2*k3+k4)
        ti=ti+h
    return y

def TGA_single(A,E,alpha):
    #Extra model parametes
    Tf=600          # Final temperature
    R=8.314         # Ideal gas constant
    parameters=[A,E,alpha,R]
    #define the initial conditions
    m1=100          #initial mass
    m0=0.85*m1      #moistureless mass
    W0=0.45*m1     #initial mass of wustite
    T0=273          #initial temperature
    
    # deifne the time scale
    t0=0            #Initial time
    h=0.001          #step size
    tf=Tf/alpha     # Final time
    n=math.ceil((tf-t0)/h)  #number of steps
    tf=t0+h*n    
 
    sol=RK4(dydt_single,A,E,alpha,R,IC=[W0,T0],t0=0,h=h,n=n)    #Calculate the solution
    
    mass=[]
    
    datapoints=60
    for i in range(datapoints):
        mass.append(sol[0,int(10000/alpha*i)])
    
    
    return mass

def thinSample(sample,k):
    n=sample.shape[0]
    thin=pd.DataFrame()
    for i in range(n):
        if i%k == 0:
            thin=thin.append(sample.loc[i])
    return thin


def weights(x,data=np.zeros(60),alpha=20):
    A=x.A
    E=x.E
    sigma=x.sigma
    sol=np.array(TGA_single(pow(10,A),pow(10,E),alpha))
    w=scist.multivariate_normal.logpdf(data,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma))
    return w

def logProb(weight):
    n=len(weight)
    w=scisp.logsumexp(weight)
    W=np.array(weight)-w
    probs=np.zeros(n)
    for i in range(n):
        probs[i]=scisp.logsumexp(W[0:i+1])
    return probs

def merge_weights(sample,unique_sample):
    n=sample.shape[0]
    m=unique_sample.shape[0]
    sample_weights=[]
    counter=1
    for i in range(n):
        if unique_sample.iloc[counter].name==i:
            counter+=1
        sample_weights.append(unique_sample.iloc[counter-1].weights)
        if counter==m:
            break
    for k in range(n-i-1):
        sample_weights.append(unique_sample.iloc[counter-1].weights)
    sample['weights']=sample_weights
    return sample

def resample(sample,probs):
    a=scist.uniform.rvs()
    b=np.log(a)
    i=np.searchsorted(probs,b,side='right')
    return sample.iloc[i]

def MCMC_SMC(theta0,var,likelihood,prior,data,steps=10):
    p=theta0.weights+theta0.ps
    theta=[theta0.A, theta0.E,theta0.sigma]
    counter=0
    for i in range(steps):
        thetat=scist.multivariate_normal.rvs(mean=theta,cov=var)
        pt=likelihood(thetat,data)
        pc=pt-p
        beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
        if beta<pc:
            theta=thetat
            p=pt
            counter+=1
    theta=np.append(theta,counter/steps)
    return theta


def prior(*theta):
    p1=scist.uniform.logpdf(theta[0],loc=0,scale=30)
    p2=scist.norm.logpdf(theta[1],loc=0,scale=10)
    p3=scist.norm.logpdf(math.log10(theta[2]),loc=-1,scale=1)
    p=p1+p2+p3
    return p

def liklihood(x,data=np.zeros(60),alphas=[5,10,20]):
    A=x[0]
    E=x[1]
    sigma=x[2]
    w=prior(A,E,sigma)
    for i in range(len(alphas)):
        sol=np.array(TGA_single(pow(10,A),pow(10,E),alphas[i]))
        e=sol-data[i]
        w=w+sum(scist.norm.logpdf(e,scale=sigma))
    return w

num_cores=6


#True Values
Tmtrue=635        #This is the Tm for a heating rate of 10
Etrue=math.log10(8.616e4)
Atrue=log_Pre_exp(Tmtrue, 10**Etrue, 10)
sigmatr=0.05

alpha5=5
alpha10=10
alpha20=20


sol10=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue), alpha10))
noisy_TGA10=sol10+np.random.normal(0,sigmatr,len(sol10))

#Define the solution 
sol20=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),alpha20))



noisy_TGA20=sol20+np.random.normal(0,sigmatr,len(sol20))


sol5=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),alpha5))


noisy_TGA5=sol5+np.random.normal(0,sigmatr,len(sol5))

#Test sample
k=10000       #test sample size
#small_sample=sample.iloc[0:k]


resampleddf=pd.read_csv('SMC_output/resampled.csv')

variances=resampleddf[['A','E','sigma']].cov().to_numpy()*2

data=[noisy_TGA5,noisy_TGA10,noisy_TGA20]
samplelist=Parallel(n_jobs=num_cores)(delayed(MCMC_SMC)(theta, var=variances,likelihood=liklihood,prior=prior, data=data,steps=5) for theta in resampleddf.iloc[:12].itertuples(index=False))

Adjusted_sample=pd.DataFrame(samplelist)

Adjusted_sample.to_csv('SMC_output/adjusted.csv',index=False)
