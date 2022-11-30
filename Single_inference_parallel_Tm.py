#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:05:29 2020

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
        mass.append((sol[0,1000*i]))
    
    
    return mass


# Define the true parameters.
Tmtrue=635
Etrue=math.log10(8.616e4)
Atrue=log_Pre_exp(Tmtrue, 10**Etrue, 10)

Ltrue=1/np.sqrt(10**np.array(Atrue)*np.exp(10**np.array(Etrue)/(300*8.314))*10**np.array(Etrue)/(300**2*8.314))
logLtrue=math.log10(Ltrue)+10
alpha=10

#Define the solution 
sol=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),alpha))

#Define the true noise parameter
sigmatr=0.05

noisy_TGA=sol+np.random.normal(0,sigmatr,len(sol))

plt.plot(noisy_TGA)

    
A=[]
E=[]
Tm=[]

sigma=[]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.uniform.logpdf(theta[2],loc=600,scale=100)
    p2=scist.norm.logpdf(theta[1],loc=0,scale=10)
    p3=scist.norm.pdf(theta[3],loc=-1.5,scale=1)
    p=p1+p2+p3
    return p

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])

    

lengths=[0]
def f(j):
    sample_size=30000
    A=[]
    E=[]
    Tm=[]
    sigma=[]
    acceptance=[1]
    ps=[]
    betas=[]
    counter=1
    
    np.random.seed(seed=10*j+1)
    E.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm.append(scist.uniform.rvs(loc=600,scale=100))
    A.append(log_Pre_exp(Tm[-1],10**E[-1],alpha))
    sigma.append(np.log10(0.05))
    sol=np.array(TGA_single(math.pow(10,A[-1]),math.pow(10,E[-1]),alpha))
    p1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*(10**sigma[-1])**2))+prior(A[0],E[0],Tm[0],sigma[0])
    ps.append(p1)
    for i in range(sample_size-1):
        Tmt=scist.norm.rvs(loc=Tm[-1],scale=0.1,size=1)[0] #log normal variation 
        Et=scist.norm.rvs(loc=E[-1],scale=0.005,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.1,size=1)[0]
        At=log_Pre_exp(Tmt,10**Et, alpha)
        sol=TGA_single(pow(10,At), pow(10,Et),alpha)
        p2=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*(10**sigmat)**2))+prior(At,Et,Tmt,sigmat)
        p=p2-p1
        beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
        betas.append(beta)
        if p>beta:
            A.append(At)
            E.append(Et)
            Tm.append(Tmt)
            sigma.append(sigmat)
            p1=p2
            ps.append(p1)
            counter+=1
        else:
            A.append(A[-1])
            E.append(E[-1])
            Tm.append(Tm[-1])
            sigma.append(sigma[-1])
            ps.append(p1)

        acceptance.append(counter/(i+2))
    col_names=['A','E','sigma','acceptance','ps','Tm']
    data=np.transpose(np.array([A,E,10**np.array(sigma),acceptance,ps,Tm]))
    output=pd.DataFrame(data,columns=col_names)
    return(output)

Single_sim = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))

def extract_data(results,col='E'):
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

E=extract_data(Single_sim,'E')
A=extract_data(Single_sim,'A')
sigma=extract_data(Single_sim,'sigma')
Tm=extract_data(Single_sim,'Tm')

L=1/np.sqrt(10**np.array(A['sample'])*np.exp(10**np.array(E['sample'])/(300*8.314))*10**np.array(E['sample'])/(300**2*8.314))
logL=np.log10(L)+10

Chain1=Single_sim[0]
Chain2=Single_sim[1]
Chain3=Single_sim[2]
Chain4=Single_sim[3]

Chain1.to_csv('Single_output/Chain1.csv',index=False)
Chain2.to_csv('Single_output/Chain2.csv',index=False)
Chain3.to_csv('Single_output/Chain3.csv',index=False)
Chain4.to_csv('Single_output/Chain4.csv',index=False)

Summary=np.zeros([10,6])
quantiles=[0.5,0,0.05,0.25,0.75,0.95,1]

def summary_statistcs(sample,True_value):
    quantiles=[0.5,0,0.05,0.25,0.75,0.95,1]
    vector=[True_value]
    vector.append(np.mean(sample))
    for k in range(7):
        vector.append(np.quantile(sample,quantiles[k]))
    vector.append(scist.percentileofscore(sample,True_value))
    return vector
    


fig1,ax1=plt.subplots()
ax1.plot(Tm['p1'],E['p1'],'.',label='Chain 1')
ax1.plot(Tm['p2'],E['p2'],'.',label='Chain 2')
ax1.plot(Tm['p3'],E['p3'],'.',label='Chain 3')
ax1.plot(Tm['p4'],E['p4'],'.',label='Chain 4')
ax1.set_xlabel('$T_m$')
ax1.set_ylabel('$log(E)$')
fig1.savefig('Single_output/trace_plots.pdf')

fig2,ax2=plt.subplots()
density=scist.gaussian_kde(A['sample'])
n,y,_ =ax2.hist(A['sample'],bins=50,density=True)
ax2.plot(y,density(y))
ax2.plot(Atrue,density(Atrue),'o')
fig2.savefig('Single_output/hist_A.pdf')

fig3,ax3=plt.subplots()
density=scist.gaussian_kde(E['sample'])
n,y,_ =ax3.hist(E['sample'],bins=50,density=True)
ax3.plot(y,density(y))
ax3.plot(Etrue,density(Etrue),'o')
fig3.savefig('Single_output/hist_E.pdf')

fig4,ax4=plt.subplots()
density=scist.gaussian_kde(sigma['sample'])
n,y,_ =ax4.hist(sigma['sample'],bins=50,density=True)
ax4.plot(y,density(y))
ax4.plot(sigmatr,density(sigmatr),'o')
fig4.savefig('Single_output/hist_sigma.pdf')

fig5,ax5=plt.subplots()
ax5.plot(Tm['sample'],E['sample'],'.')
ax5.set_xlabel('$T_m$')
ax5.set_ylabel('$log(E)$')
fig5.savefig('Single_output/joint_distribution.pdf')

fig6,ax6=plt.subplots()
density=scist.gaussian_kde(Tm['sample'])
n,y,_ =ax6.hist(Tm['sample'],bins=50,density=True)
ax6.plot(y,density(y))
ax6.plot(Tmtrue,density(Tmtrue),'o')
fig6.savefig('Single_output/hist_Tm.pdf')

fig7,ax7=plt.subplots()
density=scist.gaussian_kde(logL)
n,y,_ =ax7.hist(logL,bins=50,density=True)
ax7.plot(y,density(y))
ax7.plot(logLtrue,density(logLtrue),'o')
fig7.savefig('Single_output/hist_L.pdf')
