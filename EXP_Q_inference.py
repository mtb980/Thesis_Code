#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 08:17:20 2021

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
import random
#Set random seed for reproducibility 
np.random.seed(1)


def log_Pre_exp(Tm,E,alpha):
    R=8.314
    A=E/R/Tm*np.log10(np.exp(1))+np.log10(E/R/Tm**2*alpha)
    return A


def dydt(t, x, parameters):
    dyd=[0,0,0]
    dyd[0]=-x[0]*parameters[0]*math.exp(-parameters[2]/(parameters[5]*x[2]))
    dyd[1]=-x[1]*parameters[1]*math.exp(-parameters[3]/(parameters[5]*x[2]))
    dyd[2]=parameters[4]
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

def TGA(A1, A2, E1, E2,Q1,Q2, alpha):
    #Extra model parametesexp
    Tf=600          # Final temperature
    R=8.314         # Ideal gas constant
    parameters=[A1,A2,E1,E2,alpha,R]
    #define the initial conditions
    m0=98.36         #initial mass
    m1=0.8464*m0      #moistureless mass
    I0=0.1495*m1     # initial mass of iron
    W0=0.2942*m1     #initial mass of wustite
    T0=273          #initial temperature
    
    # deifne the time scale
    t0=0            #Initial time
    h=0.001          #step size
    tf=Tf/alpha     # Final time
    n=math.ceil((tf-t0)/h)  #number of steps
    tf=t0+h*n    
 
    sol=RK4(dydt,A1,A2,E1,E2,alpha,R,IC=[I0,W0,T0],t0=0,h=h,n=n)    #Calculate the solution
    
    mass=[]
    H1=[]
    H2=[]
    wc1=-0.376
    wc2=-0.074
    #Q1=7377.56
    #Q2=1950
    
    datapoints=43
    for i in range(datapoints):
        index=19000+700*i
        mass.append(m1-wc1*(I0-sol[0,index])-wc2*(W0-sol[1,index]))
        H1.append(Q1*A1*sol[0,index]*np.exp(-E1/(R*sol[2,index])))
        H2.append(Q2*A2*sol[1,index]*np.exp(-E2/(R*sol[2,index])))
    H=(np.array(H1)+np.array(H2))/m1/60
    
    return [mass,H,H1,H2]


m0=98.36
alpha=10
N=43

data=pd.read_csv('data/TGA_data.csv',header=0,usecols=([0,1,2,3]))
data.columns=['temp','time','DSC','FWC']
data_restrict=data[(data['time']>30) & (data['time']<60)]
data_mass=(data_restrict.values[:,3]*m0/100)[::70]
data_DSC=(data_restrict.values[:,2])[::70]
data_temp=(data_restrict.values[:,0])[::70]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.uniform.logpdf(theta[0],loc=0,scale=10)    #Prior on E1
    p2=scist.norm.logpdf(theta[1],loc=613,scale=80) #Prior on Tm1
    p3=scist.uniform.logpdf(theta[2],loc=0,scale=10000) #Prior on Q1
    p4=scist.uniform.logpdf(theta[3],loc=0,scale=10)    #Prior on E2
    p5=scist.norm.logpdf(theta[4],loc=673,scale=50) #Prior on Tm2
    p6=scist.uniform.logpdf(theta[5],loc=0,scale=10000) #Prior on Q2
    p7=scist.gamma.logpdf(theta[6],1,scale=1000)       #Prior on sigma
    p8=scist.norm.logpdf(theta[7],1,scale=1000)     #Prior on sigmaD
    p=p1+p2+p3+p4+p5+p6+p7+p8
    return p

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])

    

lengths=[0]
def f(j):
    #define sample size
    sample_size=100
    
    #data size
    N=43
    
    #Initialise the MCMC algorithm
    np.random.seed(seed=10*j+1)
    
    #Initalise parameter lists
    E1=[0]*sample_size
    Tm1=[0]*sample_size
    Q1=[0]*sample_size
    E2=[0]*sample_size
    Tm2=[0]*sample_size
    Q2=[0]*sample_size
    TauM=[0]*sample_size
    TauD=[0]*sample_size
    
    #Initalise subsequent parameter
    A1=[0]*sample_size
    A2=[0]*sample_size
    sigmaM=[0]*sample_size
    sigmaD=[0]*sample_size
    acceptance=[0]*sample_size
    
    #define Auxillary parameters
    alpha0=1
    a=alpha0+N/2
    beta0=0.001
    
    #Initialise Parameter values
    E1[0]=scist.uniform.rvs(loc=4,scale=2)
    Tm1[0]=scist.norm.rvs(loc=613,scale=80)
    Q1[0]=scist.uniform.rvs(loc=0,scale=10000)
    E2[0]=scist.uniform.rvs(loc=4,scale=2)
    Tm2[0]=scist.norm.rvs(loc=673,scale=50)
    Q2[0]=scist.uniform.rvs(loc=0,scale=10000)
    TauM[0]=scist.gamma.rvs(a,scale=1/beta0)
    TauD[0]=scist.gamma.rvs(a,scale=1/beta0)

    
    A1[0]=log_Pre_exp(Tm1[0],10**E1[0],alpha)
    A2[0]=log_Pre_exp(Tm2[0],10**E2[0],alpha)
    sigmaM[0]=1/np.sqrt(TauM[0])
    sigmaD[0]=1/np.sqrt(TauD[0])
    
    #Solve the ODE equation
    sol=np.array(TGA(math.pow(10,A1[0]),pow(10,A2[0]),math.pow(10,E1[0]),pow(10,E2[0]),Q1[0],Q2[0],alpha))
    
    
    #Determine Residuals
    eM=sol[0]-data_mass
    eD=sol[1]-data_DSC
    
    #calculate betas
    betaMi=beta0+np.linalg.norm(eM)**2/2
    betaDi=beta0+np.linalg.norm(eD)**2/2
    
    #Evaluate the Liklihood
    p1_M=sum(scist.norm.logpdf(eM,scale=sigmaM[0]))
    p1_P=prior(E1[0],Tm1[0],Q1[0],E2[0],Tm2[0],Q2[0],TauM[0],TauD[0])
    p1_D=sum(scist.norm.logpdf(eD,scale=sigmaD[0]))
    p1=p1_M+p1_D+p1_P
    
    #initalise some miscellaneous parameters
    acceptance[0]=1
    counter=1
    scale=1
    pf=0.1*scale
    for i in range(1,sample_size):
        """
        if i>sample_size/2:
            pf=1*scale
        elif i>sample_size/4:
            pf=0.1*scale
        """
        #Propose new values for the parameters from random walks
        Tm1t=scist.norm.rvs(loc=Tm1[i-1],scale=0.1/pf)
        E1t=scist.norm.rvs(loc=E1[i-1],scale=0.01/pf)
        Q1t=scist.norm.rvs(loc=Q1[i-1],scale=30/pf)
        Tm2t=scist.norm.rvs(loc=Tm2[i-1],scale=0.1/pf)
        E2t=scist.norm.rvs(loc=E2[i-1],scale=0.01/pf)
        Q2t=scist.norm.rvs(loc=Q2[i-1],scale=15/pf)
        
        #Propose new Tau from proposal distribution
        TauMt=scist.gamma.rvs(a,scale=1/betaMi)
        TauDt=scist.gamma.rvs(a,scale=1/betaDi)
        
        #Calculate additional parameters
        A1t=log_Pre_exp(Tm1t,10**E1t, alpha)
        A2t=log_Pre_exp(Tm2t,10**E2t, alpha)
        sigmaMt=1/np.sqrt(TauMt)
        sigmaDt=1/np.sqrt(TauDt)
        
        #Solve the Differential Equation
        sol=TGA(pow(10,A1t),pow(10,A2t), pow(10,E1t),pow(10,E2t),Q1t,Q2t,alpha)
        
        #Calculate Residuals
        eM=sol[0]-data_mass
        eD=sol[1]-data_DSC
        
        #Calculate new Betas
        betaMt=beta0+np.linalg.norm(eM)**2/2
        betaDt=beta0+np.linalg.norm(eD)**2/2
        
        #Evaluate the Log-Liklihood
        p2_M=sum(scist.norm.logpdf(eM,scale=sigmaMt))
        p2_P=prior(E1t,Tm1t,Q1t,E2t,Tm2t,Q2t,TauMt,TauDt)
        p2_D=sum(scist.norm.logpdf(eD,scale=sigmaDt))
        p2=p2_M+p2_D+p2_P
        
        #Evaluate the ratio of proposals
        q_M=scist.gamma.logpdf(TauM[i-1],a,scale=1/betaMt)-scist.gamma.logpdf(TauMt,a,scale=1/betaMi)
        q_D=scist.gamma.logpdf(TauD[i-1],a,scale=1/betaDt)-scist.gamma.logpdf(TauDt,a,scale=1/betaDi)
        q=q_M+q_D
        
        #Evaluate the acceptance probability
        p=p2-p1+q
        #Determine the acceptance parameter
        beta=np.log(scist.uniform.rvs(loc=0,scale=1))
        if p>beta:
            #Accept the new Values
            E1[i]=E1t
            Tm1[i]=Tm1t
            Q1[i]=Q1t
            E2[i]=E2t
            Tm2[i]=Tm2t
            Q2[i]=Q2t
            TauM[i]=TauMt
            TauD[i]=TauDt
            
            A1[i]=A1t
            A2[i]=A2t
            sigmaM[i]=sigmaMt
            sigmaD[i]=sigmaDt
            
            betaMi=betaMt
            betaDi=betaDt
            
            p1=p2
            counter+=1
        else:
            #Set to previous values
            E1[i]=E1[i-1]
            Tm1[i]=Tm1[i-1]
            Q1[i]=Q1[i-1]
            E2[i]=E2[i-1]
            Tm2[i]=Tm2[i-1]
            Q2[i]=Q2[i-1]
            TauM[i]=TauM[i-1]
            TauD[i]=TauD[i-1]
            
            A1[i]=A1[i-1]
            A2[i]=A2[i-1]
            sigmaM[i]=sigmaM[i-1]
            sigmaD[i]=sigmaD[i-1]
        
        #Set Miscellaneous Parameters
        acceptance[i]=(counter/(i+2))
    col_names=['A1','E1','sigmaM','acceptance','Tm1','A2','E2','Tm2','sigmaD','Q1','Q2']
    data=np.transpose(np.array([A1,E1,sigmaM,acceptance,Tm1,A2,E2,Tm2,sigmaD,Q1,Q2]))
    return(pd.DataFrame(data,columns=col_names))

exp_results = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))


Chain1=exp_results[0]
Chain2=exp_results[1]
Chain3=exp_results[2]
Chain4=exp_results[3]

Chain1.to_csv('EXP_Q_output/Chain1.csv',index=False)
Chain2.to_csv('EXP_Q_output/Chain2.csv',index=False)
Chain3.to_csv('EXP_Q_output/Chain3.csv',index=False)
Chain4.to_csv('EXP_Q_output/Chain4.csv',index=False)

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

E1=extract_data(exp_results,'E1')
A1=extract_data(exp_results,'A1')
E2=extract_data(exp_results,'E2')
A2=extract_data(exp_results,'A2')
Tm1=extract_data(exp_results,'Tm1')
Tm2=extract_data(exp_results,'Tm2')
sigmaM=extract_data(exp_results,'sigmaM')
Q1=extract_data(exp_results,'Q1')
Q2=extract_data(exp_results,'Q2')
acceptance=extract_data(exp_results,'acceptance')
sigmaD=extract_data(exp_results,'sigmaD')


random.seed(1)
i=random.randint(0,len(A1['sample']))
sample_sol=TGA(10**A1['sample'][i],10**A2['sample'][i],10**E1['sample'][i],10**E2['sample'][i],Q1['sample'][i],Q2['sample'][i],alpha)
"""
fig1,ax1=plt.subplots()
ax1.plot(data_temp+273,sample_sol[0])
ax1.plot(data_temp+273,data_mass)
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Mass (mg)')
ax1.legend(['Simulated Result','Experimental Result'])
fig1.savefig('EXP_Q_output/FWC.pdf')

fig2,ax2=plt.subplots()
ax2.plot(data_temp+273,sample_sol[1])
ax2.plot(data_temp+273,data_DSC)
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('DSC')
ax2.legend(['Simulated Result','Experimental Result'])
fig2.savefig('EXP_Q_output/DSC.pdf')
"""
fig2,ax2=plt.subplots()
fig1,ax1=plt.subplots()
for j in range(10):
    i=random.randint(0,4000)
    sample_sol=TGA(10**A1['sample'][i],10**A2['sample'][i],10**E1['sample'][i],10**E2['sample'][i],Q1['sample'][i],Q2['sample'][i],alpha)
    ax2.plot(data_temp+273,sample_sol[1])
    ax1.plot(data_temp+273,sample_sol[0])

ax1.plot(data_temp+273,data_mass,'.')
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Mass (mg)')
fig1.savefig('EXP_Q_output/FWC.pdf')

ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('DSC')
ax2.plot(data_temp+273,data_DSC,'.')
fig2.savefig('EXP_Q_output/DSC.pdf')

x=list(zip(A1['sample'],E1['sample'],Q1['sample'],Tm1['sample'],A2['sample'],E2['sample'],Q2['sample'],Tm2['sample'],sigma['sample'],sigmaD['sample']))

w=ps['sample']
W=w-scisp.logsumexp(w)
#w_=np.exp(np.array(w)-max(w))
#W=w_/sum(w_)
probs=[]
for i in range(len(W)):
    probs.append(scisp.logsumexp(W[:(i+1)]))

def resample(x):
    a=scist.uniform.rvs()
    b=np.log(a)
    i=np.searchsorted(probs,b,side='right')
    return x[i]

Xnew=pd.DataFrame(Parallel(n_jobs=4)(delayed(resample)(x) for i in range(len(x))),columns=['A1','E1','Q1','Tm1','A2','E2','Q2','Tm2' ,'sigma','sigmaD'])
Xnew.to_csv('EXP_Q_output/importance_sampled.csv',index=False)

density=scist.gaussian_kde(Xnew['Tm1'])
n,y,_ =plt.hist(Xnew['Tm1'],bins=20,density=True)
plt.plot(y,density(y))
density=scist.gaussian_kde(Tm1['sample'])
n,y,_ =plt.hist(Tm1['sample'],bins=100,density=True)
plt.plot(y,density(y))

L1=1/np.sqrt(np.array(Q1['sample'])*10**np.array(A1['sample'])*np.exp(10**np.array(E1['sample'])/(300*8.314))*10**np.array(E1['sample'])/(300**2*8.314))
logL1=np.log10(L1)

L2=1/np.sqrt(np.array(Q2['sample'])*10**np.array(A2['sample'])*np.exp(10**np.array(E2['sample'])/(300*8.314))*10**np.array(E2['sample'])/(300**2*8.314))
logL2=np.log10(L2)


