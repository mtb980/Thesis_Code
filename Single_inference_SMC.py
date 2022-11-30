#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:22:52 2020

@author: matt
"""


from joblib import Parallel, delayed

#import ODE_TGA as odt
import scipy.stats as scist
import scipy.special as scisp
import matplotlib.pyplot as plt
import numpy as np
import math 
import csv
import pandas as pd

#Set random seed for reproducibility 
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

def extract_data(results,col='E1'):
    p1=results[0][col].values.tolist()
    p2=results[1][col].values.tolist()
    p3=results[2][col].values.tolist()
    p4=results[3][col].values.tolist()
    chains=[p1,p2,p3,p4]
    N=min(map(len,chains))
    
    sample=p1[-1000:]+p2[-1000:]+p3[-1000:]+p4[-1000:]
    thetabars=[np.mean(p1[0:N]),np.mean(p2[0:N]),np.mean(p3[0:N]),np.mean(p4[0:N])]
    Vars=[np.var(p1[0:N],ddof=1),np.var(p2[:N],ddof=1),np.var(p3[0:N],ddof=1),np.var(p4[0:N],ddof=1)]
    B=N*np.var(thetabars,ddof=1)
    W=np.mean(Vars)
    Var=(N-1)/N*W+B/N
    rhat=np.sqrt(Var/W)
    
    d={'chains':chains,
       'p1': p1,
       'p2': p2,
       'p3': p3,
       'p4': p4,
       'sample': sample,
       'rhat':rhat}
    return d



# Define the true parameters.
Atrue=math.log10(3.17e6)
Etrue=math.log10(8.616e4)
Ltrue=1/np.sqrt(10**np.array(Atrue)*np.exp(10**np.array(Etrue)/(300*8.314))*10**np.array(Etrue)/(300**2*8.314))

alpha=10

#Define the solution 
sol=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),alpha))
sol20=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),20))
sol5=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),5))

#Define the true noise parameter
sigma=0.1
#sigma=1

noisy_TGA=sol+np.random.normal(0,sigma,len(sol))
noisy_TGA20=sol20+np.random.normal(0,sigma,len(sol20))
noisy_TGA5=sol5+np.random.normal(0,sigma,len(sol5))

plt.plot(noisy_TGA)

    


#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.norm.logpdf(theta[0],loc=math.log10(3.17e6))
    p2=scist.norm.logpdf(theta[1],loc=math.log10(8.61e4))
    p3=scist.uniform.logpdf(theta[2],loc=600,scale=100)
    p4=scist.norm.logpdf(math.log10(theta[3]),loc=-2,scale=1)
    p=p1+p2+p3+p4
    return p3+p4

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])

    

lengths=[0]
def f(j):
    A=[]
    E=[]
    Tm=[]
    sigma=[]
    acceptance=[1]
    ps=[]
    betas=[]
    np.random.seed(seed=10*j+1)
    E.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm.append(scist.uniform.rvs(loc=600,scale=100))
    A.append(log_Pre_exp(Tm[-1],10**E[-1],alpha))
    #sigma.append(0.1)
    sigma.append(0.1)
    sol=np.array(TGA_single(math.pow(10,A[-1]),math.pow(10,E[-1]),alpha))
    p1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[-1]))+prior(A[0],E[0],Tm[0],sigma[0])
    ps.append(p1)
    for i in range(30000):
        Tmt=scist.norm.rvs(loc=Tm[-1],scale=0.5,size=1)[0] #log normal variation 
        Et=scist.norm.rvs(loc=E[-1],scale=0.005,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.001,size=1)[0]
        if sigmat > 0:
            At=log_Pre_exp(Tmt,10**Et, alpha)
            sol=TGA_single(pow(10,At), pow(10,Et),alpha)
            p2=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigmat))+prior(At,Et,Tmt,sigmat)
            p=p2-p1
            beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
            betas.append(beta)
            if p>beta:
                A.append(At)
                E.append(Et)
                Tm.append(Tmt)
                sigma.append(sigmat)
                p1=p2
                acceptance.append((len(A)-lengths[-1])/(i+2))
                ps.append(p1)
    col_names=['A','E','sigma','acceptance','ps','Tm']
    data=np.transpose(np.array([A,E,sigma,acceptance,ps,Tm]))
    output=pd.DataFrame(data,columns=col_names)
    #pd.DataFrame(data,columns=col_names)
    return(output)

simulated_results = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))

E=extract_data(simulated_results,'E')
A=extract_data(simulated_results,'A')
Tm=extract_data(simulated_results,'Tm')
sigma=extract_data(simulated_results,'sigma')

Chain1=simulated_results[0]
Chain2=simulated_results[1]
Chain3=simulated_results[2]
Chain4=simulated_results[3]

Chain1.to_csv('SMC_output/Chain1.csv',index=False)
Chain2.to_csv('SMC_output/Chain2.csv',index=False)
Chain3.to_csv('SMC_output/Chain3.csv',index=False)
Chain4.to_csv('SMC_output/Chain4.csv',index=False)

L=1/np.sqrt(10**np.array(A['sample'])*np.exp(10**np.array(E['sample'])/(300*8.314))*10**np.array(E['sample'])/(300**2*8.314))
logL=np.log10(L)


x=list(zip(A['sample'],E['sample'],sigma['sample']))

def weight(x):
    A=x[0]
    E=x[1]
    sigma=x[2]
    alpha=20
    sol20=np.array(TGA_single(pow(10,A),pow(10,E),alpha))
    w=scist.multivariate_normal.logpdf(noisy_TGA20,mean=np.array(sol20),cov=np.diag(np.ones(len(sol))*sigma))
    return w

def weight5(x):
    A=x[0]
    E=x[1]
    sigma=x[2]
    alpha=5
    sol5=np.array(TGA_single(pow(10,A),pow(10,E),alpha))
    w=scist.multivariate_normal.logpdf(noisy_TGA5,mean=np.array(sol20),cov=np.diag(np.ones(len(sol))*sigma))
    return w



w = Parallel(n_jobs=4)(delayed(weight)(x[i]) for i in range(len(x)))

W=w-scisp.logsumexp(w)
#w_=np.exp(np.array(w)-max(w))
#W=w_/sum(w_)
probs=[]
for i in length(W):
    probs.append(scisp.logsumexp(w[:(i+1)]))


def resample(x):
    a=scist.uniform.rvs()
    b=np.log(a)
    i=np.searchsorted(probs,b,side='right')
    return x[i]

Xnew=pd.DataFrame(Parallel(n_jobs=4)(delayed(resample)(x) for i in range(len(x))),columns=['A','E','sigma'])
Xnew.to_csv('SMC_output/New_data.csv',index=False)

density=scist.gaussian_kde(Xnew['A'])
n,y,_ =plt.hist(Xnew['A'],bins=20,density=True)
plt.plot(y,density(y))
density=scist.gaussian_kde(A['sample'])
n,y,_ =plt.hist(A['sample'],bins=20,density=True)
plt.plot(y,density(y))
plt.plot(Atrue,1,'o')

X=list(zip(Xnew['A'],Xnew['E'],Xnew['sigma']))

w = Parallel(n_jobs=4)(delayed(weight5)(X[i]) for i in range(len(x)))

w_=np.exp(np.array(w)-max(w))
W=w_/sum(w_)
probs=np.cumsum(W)

def resample(x):
    b=scist.uniform.rvs()
    i=np.searchsorted(probs,b,side='right')
    return x[i]

Xnew5=pd.DataFrame(Parallel(n_jobs=4)(delayed(resample)(X) for i in range(len(x))),columns=['A','E','sigma'])
Xnew5.to_csv('SMC_output/Another_New_data.csv',index=False)

density=scist.gaussian_kde(Xnew['A'])
n,y,_ =plt.hist(Xnew['A'],bins=20,density=True)
plt.plot(y,density(y))
density=scist.gaussian_kde(A['sample'])
n,y,_ =plt.hist(A['sample'],bins=20,density=True)
plt.plot(y,density(y))
density=scist.gaussian_kde(Xnew5['A'])
n,y,_ =plt.hist(Xnew5['A'],bins=20,density=True)
plt.plot(y,density(y))
plt.plot(Atrue,1,'o')