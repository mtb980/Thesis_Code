
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

def TGA(A1, A2, E1, E2, alpha):
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
    wc1=-0.2863
    wc2=-0.2236*0.5
    
    datapoints=60
    for i in range(datapoints):
        mass.append(m0-wc1*(I0-sol[0,int(10000/alpha*i)])-wc2*(W0-sol[1,int(10000/alpha*i)]))
    
    
    return mass


# Define the true parameters.
A1true=math.log10(3.5e7)
E1true=math.log10(8.6e4)
A2true=math.log10(7.14e17)
E2true=math.log10(2.6e5)
#T1true=635
#T2true=691

L1true=1/np.sqrt(10**np.array(A1true)*np.exp(10**np.array(E1true)/(300*8.314))*10**np.array(E1true)/(300**2*8.314))

alpha=10

#Define the solution 
sol=np.array(TGA(pow(10,A1true),pow(10,A2true),pow(10,E1true),pow(10,E2true),alpha))
N=len(sol)
#Define the true noise parameter
sigma=0.05

noisy_TGA=sol+np.random.normal(0,sigma,len(sol))

plt.plot(noisy_TGA)

    
A1=[]
E1=[]
Tm1=[]
A2=[]
E2=[]
Tm2=[]

sigma=[]

#sol=np.array(odt.TGA_single(math.pow(10,A[0]),math.pow(10,E[0]),alpha))

def prior(*theta):
    p1=scist.norm.logpdf(theta[0],loc=math.log10(3.17e6))
    p2=scist.norm.logpdf(theta[1],loc=math.log10(8.61e4))
    p3=scist.uniform.logpdf(theta[2],loc=500,scale=100)
    p4=scist.norm.logpdf(math.log10(theta[3]),loc=-2,scale=1)
    p5=scist.uniform.logpdf(theta[6],loc=700,scale=200)
    p=p1+p2+p3+p4+p5
    return p3+p4+p5

#p1=scist.multivariate_normal.pdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(len(sol))*sigma[0]))*prior(A[0],E[0],sigma[0])
acceptance=[]
ps=[]
betas=[]
    

lengths=[0]
def f(j):
    np.random.seed(seed=10*j+1)
    E1.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm1.append(scist.uniform.rvs(loc=500,scale=100))
    A1.append(log_Pre_exp(Tm1[-1],10**E1[-1],alpha))
    E2.append(scist.uniform.rvs(loc=4,scale=2,size=1)[0])
    Tm2.append(scist.uniform.rvs(loc=700,scale=100))
    A2.append(log_Pre_exp(Tm2[-1],10**E2[-1],alpha))
    sigma.append(0.05)
    sol=np.array(TGA(math.pow(10,A1[-1]),pow(10,A2[-1]),math.pow(10,E1[-1]),pow(10,E2[-1]),alpha))
    p1=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(N)*sigma[-1]**2))+prior(A1[0],E1[0],Tm1[0],sigma[0],A2[0],E2[0],Tm2[0])
    ps.append(p1)
    for i in range(30000):
        Tm1t=scist.norm.rvs(loc=Tm1[-1],scale=0.5,size=1)[0] #log normal variation 
        E1t=scist.norm.rvs(loc=E1[-1],scale=0.005,size=1)[0]
        sigmat=scist.norm.rvs(loc=sigma[-1],scale=0.001,size=1)[0]
        A1t=log_Pre_exp(Tm1t,10**E1t, alpha)
        Tm2t=scist.norm.rvs(loc=Tm2[-1],scale=0.5,size=1)[0] #log normal variation 
        E2t=scist.norm.rvs(loc=E2[-1],scale=0.005,size=1)[0]
        A2t=log_Pre_exp(Tm2t,10**E2t, alpha)
        sol=TGA(pow(10,A1t),pow(10,A2t), pow(10,E1t),pow(10,E2t),alpha)
        if sigmat > 0:
            p2=scist.multivariate_normal.logpdf(noisy_TGA,mean=np.array(sol),cov=np.diag(np.ones(N)*sigmat**2))+prior(A1t,E1t,Tm1t,sigmat,A2t,E2t,Tm2t)
            p=p2-p1
            beta=np.log(scist.uniform.rvs(loc=0,scale=1,size=1)[0])
            betas.append(beta)
            if p>beta:
                A1.append(A1t)
                E1.append(E1t)
                Tm1.append(Tm1t)
                A2.append(A2t)
                E2.append(E2t)
                Tm2.append(Tm2t)
                sigma.append(sigmat)
                p1=p2
                acceptance.append((len(A1)-lengths[-1])/(i+2))
                ps.append(p1)
    
    return(np.array([A1,E1,sigma,acceptance,ps,Tm1,A2,E2,Tm2]))

processed_list = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))

E1p1=processed_list[0][1]
E1p2=processed_list[1][1]
E1p3=processed_list[2][1]
E1p4=processed_list[3][1]

E1=E1p1[-1000:]+E1p2[-1000:]+E1p3[-1000:]+E1p4[-1000:]
E1power=10**np.array(E1)
E1thetabars=[np.mean(E1p1[0:2500]),np.mean(E1p2[0:2500]),np.mean(E1p3[0:2500]),np.mean(E1p4[0:2500])]
E1vars=[np.var(E1p1[0:2500],ddof=1),np.var(E1p2[:2500],ddof=1),np.var(E1p3[0:2500],ddof=1),np.var(E1p4[0:2500],ddof=1)]
N=2500
E1B=N*np.var(E1thetabars,ddof=1)
E1W=np.mean(E1vars)
VarE1=(N-1)/N*E1W+E1B/N
rhatE1=np.sqrt(VarE1/E1W)


E2p1=processed_list[0][7]
E2p2=processed_list[1][7]
E2p3=processed_list[2][7]
E2p4=processed_list[3][7]

E2=E2p1[-1000:]+E2p2[-1000:]+E2p3[-1000:]+E2p4[-1000:]
E1power=10**np.array(E1)
E1thetabars=[np.mean(E1p1[0:2500]),np.mean(E1p2[0:2500]),np.mean(E1p3[0:2500]),np.mean(E1p4[0:2500])]
E1vars=[np.var(E1p1[0:2500],ddof=1),np.var(E1p2[:2500],ddof=1),np.var(E1p3[0:2500],ddof=1),np.var(E1p4[0:2500],ddof=1)]
N=2500
E1B=N*np.var(E1thetabars,ddof=1)
E1W=np.mean(E1vars)
VarE1=(N-1)/N*E1W+E1B/N
rhatE1=np.sqrt(VarE1/E1W)

T1p1=processed_list[0][5]
T1p2=processed_list[1][5]
T1p3=processed_list[2][5]
T1p4=processed_list[3][5]
Tm1=T1p1[-1000:]+T1p2[-1000:]+T1p3[-1000:]+T1p4[-1000:]
T1thetabars=[np.mean(T1p1[0:2500]),np.mean(T1p2[0:2500]),np.mean(T1p3[0:2500]),np.mean(T1p4[0:2500])]
T1thetabar=np.mean(Tm1)
T1vars=[np.var(T1p1[0:2500],ddof=1),np.var(T1p2[:2500],ddof=1),np.var(T1p3[0:2500],ddof=1),np.var(T1p4[0:2500],ddof=1)]
N=2500
T1B=N*np.var(T1thetabars,ddof=1)
T1W=np.mean(T1vars)
VarT1=(N-1)/N*T1W+T1B/N
rhatT1=np.sqrt(VarT1/T1W)

T2p1=processed_list[0][8]
T2p2=processed_list[1][8]
T2p3=processed_list[2][8]
T2p4=processed_list[3][8]
Tm2=T2p1[-1000:]+T2p2[-1000:]+T2p3[-1000:]+T2p4[-1000:]
T1thetabars=[np.mean(T1p1[0:2500]),np.mean(T1p2[0:2500]),np.mean(T1p3[0:2500]),np.mean(T1p4[0:2500])]
T1thetabar=np.mean(Tm1)
T1vars=[np.var(T1p1[0:2500],ddof=1),np.var(T1p2[:2500],ddof=1),np.var(T1p3[0:2500],ddof=1),np.var(T1p4[0:2500],ddof=1)]
N=2500
T1B=N*np.var(T1thetabars,ddof=1)
T1W=np.mean(T1vars)
VarT1=(N-1)/N*T1W+T1B/N
rhatT1=np.sqrt(VarT1/T1W)


A1p1=processed_list[0][0]
A1p2=processed_list[1][0]
A1p3=processed_list[2][0]
A1p4=processed_list[3][0]

A1=A1p1[-1000:]+A1p2[-1000:]+A1p3[-1000:]+A1p4[-1000:]
A1power=10**np.array(A1)
A1thetabars=[np.mean(A1p1[-1000:]),np.mean(A1p2[-1000:]),np.mean(A1p3[-1000:]),np.mean(A1p4[-1000:])]
A1thetabar=np.mean(A1)
A1vars=[np.var(A1p1[-1000:],ddof=1),np.var(A1p2[-1000:],ddof=1),np.var(A1p3[-1000:],ddof=1),np.var(A1p4[-1000:],ddof=1)]
N=1000
A1B=N*np.var(A1thetabars,ddof=1)
A1W=np.mean(A1vars)
VarA1=(N-1)/N*A1W+A1B/N
rhatA1=np.sqrt(VarA1/A1W)

A2p1=processed_list[0][6]
A2p2=processed_list[1][6]
A2p3=processed_list[2][6]
A2p4=processed_list[3][6]

A2=A2p1[-1000:]+A2p2[-1000:]+A2p3[-1000:]+A2p4[-1000:]
A2power=10**np.array(A2)
A2thetabars=[np.mean(A2p1[-1000:]),np.mean(A2p2[-1000:]),np.mean(A2p3[-1000:]),np.mean(A2p4[-1000:])]
A2thetabar=np.mean(A2)
A2vars=[np.var(A2p1[-1000:],ddof=1),np.var(A2p2[-1000:],ddof=1),np.var(A2p3[-1000:],ddof=1),np.var(A2p4[-1000:],ddof=1)]
N=1000
A2B=N*np.var(A2thetabars,ddof=1)
A2W=np.mean(A2vars)
VarA2=(N-1)/N*A2W+A2B/N
rhatA2=np.sqrt(VarA2/A2W)

sp1=processed_list[0][2]
sp2=processed_list[1][2]
sp3=processed_list[2][2]
sp4=processed_list[3][2]

sigma=sp1[-1000:]+sp2[-1000:]+sp3[-1000:]+sp4[-1000:]
Sthetabars=[np.mean(sp1[0:2500]),np.mean(sp2[0:2500]),np.mean(sp3[0:2500]),np.mean(sp4[0:2500])]
Sthetabar=np.mean(sigma)
Svars=[np.var(sp1[0:2500],ddof=1),np.var(sp2[:2500],ddof=1),np.var(sp3[0:2500],ddof=1),np.var(sp4[0:2500],ddof=1)]
N=2500
SB=N*np.var(Sthetabars,ddof=1)
SW=np.mean(Svars)
VarS=(N-1)/N*SW+SB/N
rhatS=np.sqrt(VarS/SW)