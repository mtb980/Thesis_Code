#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:58:57 2021

@author: mtb980
"""
from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
import pandas as pd
import scipy.stats as scist
import load_data as ld
import random

def fenicsSolver(X,dimension=1,L=10,Time=1,Tig=800,Th=0,loc=1,proportion=0.1,conversion=[0,0],save_sim=False,plotmaxuvect=False,reactnum=0):

	#Estimated Parameters
	A1=10**X[0]
	E1=10**X[1]
	Q1=10**X[2]
	A2=10**X[3]
	E2=10**X[4]
	Q2=10**X[5]
	
	if reactnum==1:
		A2=0
		Q2=0
	elif reactnum==2:
		A1=0
		Q1=0
	elif reactnum==3:
		A1=0
		Q1=0
		A2=0
		Q2=0
	#Model Parameters
	k=80					#W/m/K
	alpha=200				#m^2/year
	Ta=290
	R=8.314	
	M0=7.874e6

	#Scaled Parameters
	epsilon1=R*Ta/E1
	delta1 = L**2*Q1*A1*np.exp(-1/epsilon1)/epsilon1/Ta/k*M0/60
	epsilon2=R*Ta/E2
	delta2 = L**2*Q2*A2*np.exp(-1/epsilon2)/epsilon1/Ta/k*M0/60
	epsilon12=epsilon1/epsilon2
	
	uig=(Tig-Ta)*E1/R/Ta**2
	
	#Consumption Parameters
	I0=0.11
	W0=0.43
	deltaW=L**2/alpha*A1*np.exp(-1/epsilon1)*60*24*365
	deltaI=L**2/alpha*A2*np.exp(-1/epsilon2)*60*24*365
	
	#Time Parameters
	Tf=Time*alpha/L**2
	num_steps = 1000     # number of time steps
	dt = Tf / num_steps # time step size
	
	#hotspot parameters
	if loc > (1-proportion):
		loc=1-proportion
	elif loc <-1+proportion:
		loc=proportion-1
	uh=Th*E1/R/Ta**2
	Ic=conversion[0]
	Wc=conversion[0]

	# Create mesh and define function space
	nx = ny = 16
	meshes={
		1: IntervalMesh(128,-1,1),
		2: RectangleMesh(Point(-1,-1),Point(1,1),nx, ny),
		3: BoxMesh(Point(-1,-1,-1),Point(1,1,1),8,8,8)
		}
	mesh=meshes.get(dimension,IntervalMesh(32,0,1))
	V = FunctionSpace(mesh, 'P', 2)

	# Define boundary condition
	u_0=Expression('0+uh*(abs(x[0]-L)<(proportion))',degree=2,uh=uh,proportion=proportion,L=loc)
	#u_02 = Expression('Oa', degree=1,Oa=Oa)
	#u_03 = Expression('1-Ic*(abs(x[0]-L)<(proportion))',degree=1,Ic=Ic,proportion=proportion,L=loc)

	
	u_D = Expression('0',degree=2)
	u_I0 = Expression('I0*(1-Ic*(abs(x[0]-L)<(proportion)))',degree=2,I0=I0,Ic=Ic,proportion=proportion,L=loc)
	u_W0 = Expression('W0*(1-Ic*(abs(x[0]-L)<(proportion)))',degree=2,W0=W0,Ic=Wc,proportion=proportion,L=loc)
	
	bc = DirichletBC(V, u_D, boundary)

	# Define initial value
	
	u_n = project(u_0, V)
	I_n = project(u_I0, V)
	W_n = project(u_W0, V)
	# Define variational problem
	u = TrialFunction(V)
	I = TrialFunction(V)
	W = TrialFunction(V)
	v = TestFunction(V)
	#f=Constant(1)
	f1 = Expression('exp(u_n/(1+epsilon1*u_n))*delta1*W_n',degree=1, delta1=delta1,u_n=u_n,epsilon1=epsilon1,W_n=W_n)
	f2 = Expression('exp((epsilon12)*u_n/(1+epsilon1*u_n))*delta2*I_n',degree=1, delta2=delta2,u_n=u_n,epsilon1=epsilon1,epsilon12=epsilon12,I_n=I_n)
#	f1 = Expression('delta1/deltaI*(W_n1-W_n)',degree=1,delta1=delta1,deltaI=deltaI,W_n1=w_n,w_n=w_n)
	#fW = Expression('exp(u_n/(1+epsilonI*u_n))*deltaI',degree=1, deltaW=deltaW,u_n=u_n,epsilon1=epsilon1)
	#fI = Expression('exp((epsilon12)*u_n/(1+epsilon1*u_n))*delta2',degree=1, deltaI=deltaI,u_n=u_n,epsilon1=epsilon1,epsilon12=epsilon12)
	
	F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n+dt*(f1+f2))*v*dx

	a1, L1 = lhs(F), rhs(F)
	ubigs={
		1:np.zeros([257,num_steps]),
		2:np.zeros([289,num_steps]),
		3:np.zeros([729,num_steps])
		}
	ubig=np.zeros([len(u_n.vector().get_local()),1000])
	if save_sim:
		vtkfile=File('output/Temperature.pvd')
		#vtkfile2=File('output/Oxygen.pvd')
		vtkfile3=File('output/Iron.pvd')
	
	maxuvect=[]
	# Time-stepping
	u = Function(V)
	t = 0
	for n in range(num_steps):
		
		# Update current time
		t += dt
		f1.u_n=u_n
		f2.u_n=u_n
		 
		unarray=u_n.vector().get_local()
		ubig[:,n]=unarray
		fIarray=np.exp(epsilon12*unarray/(1+epsilon1*unarray))*deltaI*dt
		fWarray=np.exp(unarray/(1+epsilon1*unarray))*deltaW*dt

		# Compute solution
		#solve(a3 == L3, u_3, solver_parameters={'linear_solver':'gmres','preconditioner':'ilu'})
		Inarray=I_n.vector().get_local()
		Inarray=Inarray/(1+fIarray)
		#u3array=u_3.vector().get_local()
		Inarray[Inarray<0]=0
		I_n.vector().set_local(Inarray)
		#I_n.assign(I_n)
		
		
		Wnarray=W_n.vector().get_local()
		Wnarray=Wnarray/(1+fWarray)
		#u3array=u_3.vector().get_local()
		Wnarray[Wnarray<0]=0
		W_n.vector().set_local(Wnarray)
		#W_n.assign(W_n)
		
		
		
		
		solve(a1 == L1, u,bc, solver_parameters={'linear_solver':'gmres','preconditioner':'ilu'})
		u_n.assign(u)
		
		f1.W_n=W_n
		f2.I_n=I_n
		
		# Plot solution
		#plot(u)
		if save_sim:
			if n % 10 == 0:
				vtkfile<< (u,t)
		maxu=np.max(u.compute_vertex_values(mesh))
		maxuvect.append(maxu)
		if maxu>uig:
			ignition=3
			#break
		#if n==10:
			#plot(u)
		#	plot(W_n)
			#plot(I_n)

	if maxu>uig:
		ignition=3
	elif np.median(Inarray)<(I0*(1-Ic*proportion))/2:
		ignition=2
	elif np.median(Wnarray)<W0*(1-Wc*proportion)/2:
		ignition=1
	else:
		ignition=0
			
	#plt.plot(maxuvect)
	if plotmaxuvect:
		fig,ax=plt.subplots()
		tvect=np.linspace(0,t*L**2/alpha,n+1)
		ax.plot(tvect,epsilon1*Ta*np.array(maxuvect)+Ta-273)
		ax.set_xlabel('Time (Years)')
		ax.set_ylabel('Maximum Temperature ($^o$C)')
		fig.savefig('maxtemps/consmaxuvect'+str(L)+str(reactnum)+'.pdf',bbox_inches='tight')
	return [ubig,ignition,I_n,W_n]

def criticalL(X,Th=0,loc=1,proportion=0,conversion=[0,0],state=3,dim=1):
	#dim =3
	logL1=-3
	logL2=3
	output=10**logL2
	sol=fenicsSolver(X,L=10**logL2,Th=Th,loc=loc,proportion=proportion,conversion=conversion,dimension=dim)
	if sol[1]<state:
		return 10**logL2
	sol=fenicsSolver(X,L=10**logL1,Th=Th,loc=loc,proportion=proportion,conversion=conversion,dimension=dim)
	if sol[1]>=state:
		return 10**logL1
	
	logLc=(logL2+logL1)/2
	for i in range(20):
		sol=fenicsSolver(X,L=10**logLc,Th=Th,loc=loc,proportion=proportion,conversion=conversion,dimension=dim)
		if sol[1]>=state:
			logL2=logLc
		else:
			logL1=logLc
		logLc=(logL2+logL1)/2
	return 10**logLc



def boundary(x, on_boundary):
	   return on_boundary


def assemble(data_run):
	chains=['1','2','3','4']
	folder_name=data_run+'_output'
	results=dict()
	for number in chains:
		file_name=folder_name+'/Chain'+number+'.csv'
		results['Chain'+number]=pd.read_csv(file_name)
	return results

def extract_data(results,data_key='E1'):
	chains=['1','2','3','4']
	output={}
	chainlengths=[]
	sample=[]
	for number in chains:
		output['p'+number]=results['Chain'+number][data_key].values.tolist()
		chainlengths.append(len(output['p'+number]))
		sample+=output['p'+number]
	p1=output['p1']
	p2=output['p2']
	p3=output['p3']
	p4=output['p4']
	N=min(chainlengths)
	sample=p1[-1000:]+p2[-1000:]+p3[-1000:]+p4[-1000:]
	output['sample']=sample
	return output

def generateData(results):
	column_names=list(results['Chain1'].columns)
	output={}
	for data_key in column_names:
		output[data_key]=extract_data(results,data_key=data_key)
	return output

def transformArray(results):
	X=list(zip(results['A1']['sample'],results['E1']['sample'],results['Q1']['sample'],results['A2']['sample'],results['E2']['sample'],results['Q2']['sample']))
	return X

def testCases():
	X1=[2.792,4.6463,7136,22.54,5.477,938]
	X2=[2.778,4.643,6878,19.72,5.4209,988]
	X3=[2.949,4.661,6178,16.31,5.3411,1284]
	L1=criticalL(X1)
	L2=criticalL(X2)
	L3=criticalL(X3)
	print(L1)
	print(L2)
	print(L3)

def determineCriticalLs(data_run,dim=1,state=2):
	data=ld.assemble(data_run)
	burned=ld.removeBurn(data,10000)
	resultsburn=ld.generateData(burned,burnin=True)
	X=transformArray(resultsburn)
	Lcr=Parallel(n_jobs=4)(delayed(criticalL)(X[random.randint(0,len(X))]) for i in range(500))

	fig4,ax4=plt.subplots()
	density=scist.gaussian_kde(Lcr)
	n,y,_ =ax4.hist(Lcr,bins=50,density=True)
	ax4.plot(y,density(y))
	fig4.savefig('hist_Lcr'+str(dim)+str(state)+'.pdf',bb0x_inches='tight')
	return Lcr

def criticalHotspot(X,L=4,loc=1,proportion=0.1,conversion=[0,0],state=3):
	Ta=0
	sol=fenicsSolver(X,L=L,Th=Ta,loc=loc,proportion=proportion,conversion=conversion)
	if sol[1]>=state:
		return 0
	Tb=200
	sol=fenicsSolver(X,L=L,Th=Tb,loc=loc,proportion=proportion,conversion=conversion)
	if sol[1]<state:
		return Tb
	for i in range(20):
		Tc=(Ta+Tb)/2
		sol=fenicsSolver(X,L=L,Th=Tc,loc=loc,proportion=proportion,conversion=conversion)
		if sol[1]>=state:
			Tb=Tc
		else:
			Ta=Tc
	Tc=(Ta+Tb)/2
	return Tc

def Tplots(X):
	Ls=np.linspace(7/2,9/2,33)
	states=[1,2,3]
	Tcrit=[]
	for L in Ls:
		Statecrits=[]
		for state in states:
			Statecrits.append(criticalHotspot(X,L=L,state=state,proportion=3/32,loc=29/32))
		Tcrit.append(Statecrits)
	Tcrit=np.array(Tcrit)
	fig,ax=plt.subplots()
	ax.plot(Ls,Tcrit[:,0]+18,label=r'$L_p$')
	ax.plot(Ls,Tcrit[:,1]+18,label=r'$L_c$')
	ax.legend()
	ax.set_xlabel('Length (m)')
	ax.set_ylabel('Temperature ($^o$C)')
	fig.savefig('Tcrits.pdf')
	return np.array(Tcrit)

def Tplotprop(X):
	L=9/2
	props=np.linspace(0,29/32,30)
	state=1
	Tcrit=[]
	for prop in props:
		Tcrit.append(criticalHotspot(X,L=L,state=state,proportion=prop,loc=1-prop))
	fig,ax=plt.subplots()
	ax.plot(props,Tcrit,label=r'$L_p$')
	ax.set_xlabel('Hotspot width, $h_l$')
	ax.set_ylabel('Temperature ($^o$C)')
	fig.savefig('Tcritsprop.pdf')
	return np.array(Tcrit)

def Tplotloc(X):
	L=9/2
	locs=np.linspace(0,29/32,30)
	states=[1,2]
	Tcrit=[]
	for loc in locs:
		Statecrits=[]
		for state in states:
			Statecrits.append(criticalHotspot(X,L=L,state=state,loc=loc,proportion=3/32))
		Tcrit.append(Statecrits)
	Tcrit=np.array(Tcrit)
	fig,ax=plt.subplots()
	ax.plot(locs,Tcrit[:,0]+18,label=r'$L_p$')
	ax.plot(locs,Tcrit[:,1]+18,label=r'$L_c$')
	ax.legend()
	ax.set_xlabel(r'Centre of Hotspot, $h_c$')
	ax.set_ylabel(r'Temperature ($^o$C)')
	fig.savefig('Tcritsloc.pdf')
	return np.array(Tcrit)

def Lplots(X,prop=0):
	Ts=np.linspace(0,200)
	states=[1,2]
	Lcrit=[]
	for T in Ts:
		Statecrits=[]
		for state in states:
			Statecrits.append(criticalL(X,Th=T,state=state,proportion=prop,loc=1-prop))
		Lcrit.append(Statecrits)
	Lcrit=np.array(Lcrit)
	fig,ax=plt.subplots()
	ax.plot(Ts+18,Lcrit[:,0],label=r'$L_p$')
	ax.plot(Ts+18,Lcrit[:,1],label=r'$L_c$')
	ax.legend()
	ax.set_xlabel('Temperature ($^oC$)')
	ax.set_ylabel('Length (m)')
	fig.savefig('Lcrits.pdf')
	return np.array(Lcrit)

def multiCriticalL(x):
	#dim =3
	logLp1=-3
	logLp2=3
	logLc1=-3
	logLc2=3
	logLt1=-3
	logLt2=3
	
	sol=fenicsSolver(X,L=10**logLp2,Th=Th,loc=loc,proportion=proportion,conversion=conversion)
	if sol[1]<1:
		Lp= 10**logL2
		Lc= 10**logL2
		LT= 10**logL2
		return [Lp,Lc,Lt]
	elif sol[1]<2:
		Lc= 10**logL2
		LT= 10**logL2
		Lpfound=False
	elif sol[2]<3:
		LT=10**logL2
		Lcfound=False
		Lpfound=False
	else:
		LTfound=False
		Lcfound=False
		Lpfound=False
		

	sol=fenicsSolver(X,L=10**logLp1,Th=Th,loc=loc,proportion=proportion,conversion=conversion)
	if sol[1]>=state:
		Lp= 10**logL1
	if sol[1]>=3:
		Lp= 10**logL1
		Lc= 10**logL1
		LT= 10**logL1
		return [Lp,Lc,Lt]
	elif sol[1]>=2:
		Lc= 10**logL1
		Lp= 10**logL1
		LTfound=False
	elif sol[2]>=1:
		Lp=10**logL1
		Lcfound=False
		LTfound=False
	else:
		LTfound=False
		Lcfound=False
		Lpfound=False
	
	logLc=(logL2+logL1)/2
	for i in range(20):
		sol=fenicsSolver(X,L=10**logLc,Th=Th,loc=loc,proportion=proportion,conversion=conversion)
		if sol[1]>=state:
			logL2=logLc
		else:
			logL1=logLc
		logLc=(logL2+logL1)/2
		10**logLc
	return [Lp,Lc,LT]

def criticalLallstates(x):
	criticalls=[]
	for state in range(1,4):
		for dim in range(1,4):
			criticalls.append(criticalL(x,state=state,dim=dim))
	return criticalls

def determineCriticalLsmulti(data_run,num_cores=4,samplesize=500):
	data=ld.assemble(data_run)
	burned=ld.removeBurn(data,10000)
	resultsburn=ld.generateData(burned,burnin=True)
	X=transformArray(resultsburn)
	Lcr=Parallel(n_jobs=num_cores)(delayed(criticalLallstates)(X[random.randint(0,len(X))]) for i in range(samplesize))
	return pd.DataFrame(Lcr)


if __name__=='__main__':
	#Lcr=determineCriticalLsmulti('EXP_Q',samplesize=4)
	#Lcr.to_csv('Lcrmulti.csv')
	#testCases()
	#L=4.85
	medianvals=[4.395,4.7666,3.1095,8.7515,5.0908,3.842]
	#TI=criticalHotspot(medianvals,state=2)
	#LW=criticalL(medianvals,state=1,dim=1)
	#LI=criticalL(medianvals,state=2,dim=1)
	#LT=criticalL(medianvals,state=3,dim=1)
	Lcrits=Lplots(medianvals,prop=3/32)
	Tcrits=Tplots(medianvals)
	Tcritsprop=Tplotprop(medianvals)
	Tcritsloc=Tplotloc(medianvals)
	
	#sol=fenicsSolver(medianvals,L=4.5,loc=1,proportion=0.1,plotmaxuvect=True,Th=200,reactnum=0)
	
	L=4.4
	sol0=fenicsSolver(medianvals,L=L,Th=80,loc=0,plotmaxuvect=True)
	#sol1=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=1)
	#sol2=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=2)
	
	"""
	L=4.75
	sol3=fenicsSolver(medianvals,L=L,save_sim=False,plotmaxuvect=True)
	#sol4=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=1)
	#sol5=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=2)
	
	L=4.8
	sol6=fenicsSolver(medianvals,L=L,save_sim=False,plotmaxuvect=True)
	sol7=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=1)
	sol8=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=2)
	
	L=25
	sol9=fenicsSolver(medianvals,L=L,save_sim=False,plotmaxuvect=True)
	sol10=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=1)
	#sol11=fenicsSolver(medianvals,L=L,save_sim=True,plotmaxuvect=True,reactnum=2)
	"""