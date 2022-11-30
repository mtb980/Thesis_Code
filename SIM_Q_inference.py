#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:26:41 2021

@author: matt
"""


from joblib import Parallel, delayed

#import ODE_TGA as odt
import scipy.stats as scist
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


def dydt(t, x, parameters):
    dyd=[0,0,0]
    dyd[0]=-x[0]*parameters[0]*math.exp(-parameters[2]/(parameters[5]*x[2]))
    dyd[1]=-pow(x[1],1)*parameters[1]*math.exp(-parameters[3]/(parameters[5]*x[2]))
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
    #Extra model parametes
    Tf=600          # Final temperature
    R=8.314         # Ideal gas constant
    parameters=[A1,A2,E1,E2,alpha,R]
    #define the initial conditions
    m1=105          #initial mass
    m0=0.85*m1      #moistureless mass
    I0=0.169*m0     # initial mass of iron
    W0=0.298*m0     #initial mass of wustite
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
    
    datapoints=100
    for i in range(datapoints):
        index=int(6000/alpha*i)
        mass.append(m0-wc1*(I0-sol[0,index])-wc2*(W0-sol[1,index]))
        H1.append(Q1*A1*sol[0,index]*np.exp(-E1/(R*sol[2,index])))
        H2.append(Q2*A2*sol[1,index]*np.exp(-E2/(R*sol[2,index])))
    H=(np.array(H1)+np.array(H2))/m1/60
    return [mass,H]


alpha=10

# Define the true parameters.
Tm1true=650
E1true=math.log10(8.6e4)
A1true=log_Pre_exp(Tm1true, 10**E1true, alpha)
Q1true=7300
Tm2true=700
E2true=math.log10(2.6e5)
A2true=log_Pre_exp(Tm2true, 10**E2true, alpha)
Q2true=1950

L1true=1/np.sqrt(10**np.array(A1true)*np.exp(10**np.array(E1true)/(300*8.314))*10**np.array(E1true)/(300**2*8.314))


#Define the solution 
sol=np.array(TGA(pow(10,A1true),pow(10,A2true),pow(10,E1true),pow(10,E2true),Q1true,Q2true,alpha))
N=len(sol[0])
#Define the true noise parameter
sigma=0.1

noisy_TGA=sol[0]+np.random.normal(0,sigma,N)
noisy_DSC=sol[1]+np.random.normal(0,sigma/50,N)

plt.plot(noisy_TGA)

A1=[]
E1=[]
Tm1=[]
A2=[]
E2=[]
Tm2=[]
Q1=[]
Q2=[]
sigma=[]
sigmaD=[]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.uniform.logpdf(theta[0],loc=0,scale=10)    #Prior on E1
    p2=scist.uniform.logpdf(theta[1],loc=550,scale=200) #Prior on Tm1
    p3=scist.norm.logpdf(theta[2],loc=-2,scale=1)       #Prior on sigma
    p4=scist.uniform.logpdf(theta[3],loc=0,scale=10)    #Prior on E2
    p5=scist.uniform.logpdf(theta[4],loc=max(650,theta[1]),scale=min(100,750-theta[1])) #Prior on Tm2
    p6=scist.norm.logpdf(theta[5],loc=-4,scale=0.1)     #Prior on sigmaD
    p7=scist.uniform.logpdf(theta[6],loc=0,scale=10000) #Prior on Q1
    p8=scist.uniform.logpdf(theta[7],loc=0,scale=10000) #Prior on Q2
    p=p1+p2+p3+p4+p5+p6+p7+p8
    return p

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])
acceptance=[]
ps=[]
betas=[]
    

lengths=[0]
def f(j):
    #set Sample size
    sample_size=20000
    #set random seed
    np.random.seed(seed=10*j+1)
    
    #Initialise the parameters
    E1.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm1.append(scist.uniform.rvs(loc=550,scale=100))
    A1.append(log_Pre_exp(Tm1[-1],10**E1[-1],alpha))
    Q1.append(scist.uniform.rvs(loc=5000,scale=5000))
    E2.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm2.append(scist.uniform.rvs(loc=650,scale=100))
    A2.append(log_Pre_exp(Tm2[-1],10**E2[-1],alpha))
    Q2.append(scist.uniform.rvs(loc=1000,scale=3000))
    sigma.append(np.log10(0.1))
    sigmaD.append(np.log10(0.002))
    #Initialise the solution
    sol=np.array(TGA(math.pow(10,A1[-1]),pow(10,A2[-1]),math.pow(10,E1[-1]),pow(10,E2[-1]),Q1[-1],Q2[-1],alpha))
    #Calculate the Likelihood function
    p1_1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol[0]),cov=np.diag(np.ones(N)*sigma[-1]**2))+prior(A1[0],E1[0],Tm1[0],sigma[0],A2[0],E2[0],Tm2[0],sigmaD[0],Q1[0],Q2[0])
    p1_2=scist.multivariate_normal.logpdf(noisy_DSC,mean=np.array(sol[1]),cov=np.diag(np.ones(N)*sigmaD[-1]**2))
    p1=p1_1+p1_2
    
    #initialise miscellaneous parameters
    ps.append(p1)
    acceptance.append(1)
    counter=1
    #set the proposal factors
    scale=0.1
    pf=0.05*scale
    for i in range(sample_size-1):
        if i>sample_size/2:
            pf=20*scale
        elif i>sample_size/4:
            pf=1*scale
        
        #Propose new values for the parameters
        Tm1t=scist.norm.rvs(loc=Tm1[-1],scale=1/pf,size=1)[0] #log normal variation 
        E1t=scist.norm.rvs(loc=E1[-1],scale=0.01/pf,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.01,size=1)[0]
        Q1t=scist.norm.rvs(loc=Q1[-1],scale=100/pf,size=10)[0]
        sigmaDt=scist.norm.rvs(loc=sigmaD[-1],scale=0.01,size=1)[0]
        A1t=log_Pre_exp(Tm1t,10**E1t, alpha)
        Tm2t=scist.norm.rvs(loc=Tm2[-1],scale=1/pf,size=1)[0] #log normal variation 
        E2t=scist.norm.rvs(loc=E2[-1],scale=0.01/pf,size=1)[0]
        Q2t=scist.norm.rvs(loc=Q2[-1],scale=50/pf,size=1)[0]
        A2t=log_Pre_exp(Tm2t,10**E2t, alpha)
   
        #Solve the ODE equation 
        sol=TGA(pow(10,A1t),pow(10,A2t), pow(10,E1t),pow(10,E2t),Q1t,Q2t,alpha)
        
        #Evaluate the likelihood function
        p2_1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol[0]),cov=np.diag(np.ones(N)*(10**sigmat)**2))+prior(E1t,Tm1t,sigmat,E2t,Tm2t,sigmaDt,Q1t,Q2t)
        p2_2=scist.multivariate_normal.logpdf(noisy_DSC,mean=np.array(sol[1]),cov=np.diag(np.ones(N)*(10**sigmaDt)**2))
        p2=p2_1+p2_2
        p=p2-p1
        
        beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
        if p>beta:
            #Accept the new Values
            A1.append(A1t)
            E1.append(E1t)
            Tm1.append(Tm1t)
            Q1.append(Q1t)
            A2.append(A2t)
            E2.append(E2t)
            Tm2.append(Tm2t)
            Q2.append(Q2t)
            sigma.append(sigmat)
            sigmaD.append(sigmaDt)
            p1=p2
            counter+=1
        else:
            #Set to previous values
            A1.append(A1[-1])
            E1.append(E1[-1])
            Tm1.append(Tm1[-1])
            Q1.append(Q1[-1])
            A2.append(A2[-1])
            E2.append(E2[-1])
            Tm2.append(Tm2[-1])
            Q2.append(Q2[-1])
            sigma.append(sigma[-1])
            sigmaD.append(sigmaD[-1])
        
        #Set Miscellaneous Parameters
        acceptance.append(counter/(i+2))
        ps.append(p1)
    col_names=['A1','E1','sigma','acceptance','ps','Tm1','A2','E2','Tm2','sigmaD','Q1','Q2']
    data=np.transpose(np.array([A1,E1,sigma,acceptance,ps,Tm1,A2,E2,Tm2,10**np.array(sigmaD),Q1,Q2]))
    return(pd.DataFrame(data,columns=col_names))

sim_data = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))

Chain1=sim_data[0]
Chain2=sim_data[1]
Chain3=sim_data[2]
Chain4=sim_data[3]

Chain1.to_csv('SIM_Q_output/Chain1.csv',index=False)
Chain2.to_csv('SIM_Q_output/Chain2.csv',index=False)
Chain3.to_csv('SIM_Q_output/Chain3.csv',index=False)
Chain4.to_csv('SIM_Q_output/Chain4.csv',index=False)

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

E1=extract_data(sim_data,'E1')
A1=extract_data(sim_data,'A1')
E2=extract_data(sim_data,'E2')
A2=extract_data(sim_data,'A2')
Tm1=extract_data(sim_data,'Tm1')
Tm2=extract_data(sim_data,'Tm2')
sigma=extract_data(sim_data,'sigma')
