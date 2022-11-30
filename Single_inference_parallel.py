#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:55:58 2020

@author: matt
"""
from joblib import Parallel, delayed

#import ODE_TGA as odt
import pandas as pd
import scipy.stats as scist
import matplotlib.pyplot as plt
import numpy as np
import math 
import csv

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
    W0=1*m1     #initial mass of wustite
    T0=273          #initial temperature
    
    # deifne the time scale
    t0=0            #Initial time
    h=0.001          #step size
    tf=Tf/alpha     # Final time
    n=math.ceil((tf-t0)/h)  #number of steps
    tf=t0+h*n    
 
    sol=RK4(dydt_single,A,E,alpha,R,IC=[m1,T0],t0=0,h=h,n=n)    #Calculate the solution
    
    mass=[]
    wc1=-0.2863
    wc2=-0.2236*0.5
    
    datapoints=60
    for i in range(datapoints):
        mass.append((sol[0,1000*i]))
    
    
    return mass


# Define the true parameters.
Tmtrue=635
Etrue=math.log10(8.616e4)
Atrue=log_Pre_exp(Tmtrue, Etrue,10)
Ltrue=1/np.sqrt(10**np.array(Atrue)*np.exp(10**np.array(Etrue)/(300*8.314))*10**np.array(Etrue)/(300**2*8.314))

alpha=10

#Define the solution 
sol=np.array(TGA_single(pow(10,Atrue),pow(10,Etrue),alpha))

#Define the true noise parameter
sigmatr=0.1

noisy_TGA=sol+np.random.normal(0,sigmatr,len(sol))

plt.plot(noisy_TGA)

    
A=[]
E=[]

sigma=[]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.uniform.logpdf(theta[0],loc=0,scale=30)
    p2=scist.norm.logpdf(theta[1],loc=0,scale=10)
    p3=scist.norm.logpdf(math.log10(theta[2]),loc=-1,scale=1)
    p=p1+p2+p3
    return p

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])

ps=[]
betas=[]
acceptance=[1]
    
lengths=[0]
def f(j):
    sample_size=30000
    np.random.seed(seed=10*j+1)
    E.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    A.append(scist.uniform.rvs(loc=3,scale=7))
    sigma.append(1)
    sol=np.array(TGA_single(math.pow(10,A[-1]),math.pow(10,E[-1]),alpha))
    p1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[-1]**2))+prior(A[0],E[0],sigma[0])
    ps.append(p1)
    counter=0
    
    for i in range(sample_size-1):
        At=scist.norm.rvs(loc=A[-1],scale=0.05,size=1)[0] #log normal variation 
        Et=scist.norm.rvs(loc=E[-1],scale=0.001,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.001,size=1)[0]
        if sigmat > 0:
            sol=TGA_single(pow(10,At), pow(10,Et),alpha)
            p2=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigmat**2))+prior(At,Et,sigmat)
            p=p2-p1
            beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
            betas.append(beta)
            if p>beta:
                A.append(At)
                E.append(Et)
                sigma.append(sigmat)
                p1=p2
                counter+=1
                ps.append(p1)
            else:
                A.append(A[-1])
                E.append(E[-1])
                sigma.append(sigma[-1])
                ps.append(p1)
        else:
            A.append(A[-1])
            E.append(E[-1])
            sigma.append(sigma[-1])
            ps.append(p1)
        acceptance.append(counter/(i+2))
        
    col_names=['A','E','sigma','acceptance','ps']
    data=np.transpose(np.array([A,E,sigma,acceptance,ps]))
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
acceptance=extract_data(Single_sim,'acceptance')

L=1/np.sqrt(10**np.array(A['sample'])*np.exp(10**np.array(E['sample'])/(300*8.314))*10**np.array(E['sample'])/(300**2*8.314))
logL=np.log10(L)

Chain1=Single_sim[0]
Chain2=Single_sim[1]
Chain3=Single_sim[2]
Chain4=Single_sim[3]

Chain1.to_csv('Single_base_output/Chain1.csv',index=False)
Chain2.to_csv('Single_base_output/Chain2.csv',index=False)
Chain3.to_csv('Single_base_output/Chain3.csv',index=False)
Chain4.to_csv('Single_base_output/Chain4.csv',index=False)

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
ax1.plot(A['p1'],E['p1'],'.',label='Chain 1')
ax1.plot(A['p2'],E['p2'],'.',label='Chain 2')
ax1.plot(A['p3'],E['p3'],'.',label='Chain 3')
ax1.plot(A['p4'],E['p4'],'.',label='Chain 4')
ax1.set_xlabel('$log(A)$')
ax1.set_ylabel('$log(E)$')
ax1.legend()
fig1.savefig('Single_base_output/trace_plots.pdf')

fig2,ax2=plt.subplots()
density=scist.gaussian_kde(A['sample'])
n,y,_ =ax2.hist(A['sample'],bins=50,density=True)
ax2.plot(y,density(y))
ax2.plot(Atrue,density(Atrue),'o')
ax2.set_xlabel('$log(A)$')
fig2.savefig('Single_base_output/hist_A.pdf')

fig3,ax3=plt.subplots()
density=scist.gaussian_kde(E['sample'])
n,y,_ =ax3.hist(E['sample'],bins=50,density=True)
ax3.plot(y,density(y))
ax3.plot(Etrue,density(Etrue),'o')
ax3.set_xlabel('$log(E)$')
fig3.savefig('Single_base_output/hist_E.pdf')

fig4,ax4=plt.subplots()
density=scist.gaussian_kde(sigma['sample'])
n,y,_ =ax4.hist(sigma['sample'],bins=50,density=True)
ax4.plot(y,density(y))
ax4.plot(0.1,density(0.1),'o')
ax4.set_xlabel('$\sigma$')
fig4.savefig('Single_base_output/hist_sigma.pdf')

fig5,ax5=plt.subplots()
ax5.plot(A['sample'],E['sample'],'.')
ax5.set_xlabel('$log(A)$')
ax5.set_ylabel('$log(E)$')
fig5.savefig('Single_base_output/joint_distribution.pdf')


fig6,ax6=plt.subplots()
ax6.plot(A['p1'],label='Chain 1')
ax6.plot(A['p2'],label='Chain 2')
ax6.plot(A['p3'],label='Chain 3')
ax6.plot(A['p4'],label='Chain 4')
ax6.set_ylabel('$log(A)$')
ax6.legend()
fig6.savefig('Single_base_output/trace_plot_A.pdf')

fig7,ax7=plt.subplots()
ax7.plot(E['p1'],label='Chain 1')
ax7.plot(E['p2'],label='Chain 2')
ax7.plot(E['p3'],label='Chain 3')
ax7.plot(E['p4'],label='Chain 4')
ax7.set_ylabel('$log(A)$')
ax7.legend()
fig7.savefig('Single_base_output/trace_plot_E.pdf')