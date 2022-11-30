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


def fenicsSolver(X,dimension=1,L=10,Time=1,phi=0,To=0,Tig=1000,save_sim=False,plotmaxuvect=False,reactnum=0):

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
	
	#Model Parameters
	k=80
	alpha=200
	Ta=290
	R=8.314	
	M0=7.874e6
	

	#Scaled Parameters
	epsilon1=R*Ta/E1
	delta1 = L**2*Q1*A1*np.exp(-1/epsilon1)/epsilon1/Ta/k*M0/60*0.42
	epsilon2=R*Ta/E2
	delta2 = L**2*Q2*A2*np.exp(-1/epsilon2)/epsilon2/Ta/k*M0/60*0.11
	epsilon12=epsilon1/epsilon2
	
	uig=(Tig-(Ta-273))*E1/R/Ta**2
	
	#Boundary Condition Parameters
	omega=alpha/L**2
	uo=To/epsilon1/Ta
	#phi=0 input

	#Time Parameters
	Tf=Time*alpha/(L**2)           # final time
	num_steps = 1000     # number of time steps
	dt = Tf / num_steps # time step size

	# Create mesh and define function space
	nx = ny = 25
	meshes={
		1: IntervalMesh(150,-1,1),
		2: RectangleMesh(Point(-1,-1),Point(1,1),nx, ny),
		3: BoxMesh(Point(-1,-1,-1),Point(1,1,1),10,10,10)
		}
	mesh=meshes.get(dimension,IntervalMesh(32,0,1))
	V = FunctionSpace(mesh, 'P', 2)

	# Define boundary condition
	u_D = Expression('To*sin(2*pi*t/omega+phi)',degree=2, omega=omega, t=0, To=uo,phi=phi)
	bc = DirichletBC(V, u_D, boundary)

	# Define initial value
	u_n = project(u_D, V)
	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	#f=Constant(1)
	f1 = Expression('exp(u_n/(1+epsilon1*u_n))*delta1',degree=1, delta1=delta1,u_n=u_n,epsilon1=epsilon1)
	f2 = Expression('exp((epsilon12)*u_n/(1+epsilon1*u_n))*delta2*epsilon12',degree=1, delta2=delta2,u_n=u_n,epsilon1=epsilon1,epsilon12=epsilon12)
	
	F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n+dt*(f1+f2))*v*dx

	a, l = lhs(F), rhs(F)
	
	if save_sim:
		vtkfile=File('output/Temperature'+str(To)+str(phi)+str(L)+str(reactnum)+'.pvd')
		#vtkfile2=File('output/Oxygen.pvd')
		#vtkfile3=File('output/Iron.pvd')
	
	maxuvect=[np.max(u_n.compute_vertex_values(mesh))]
	ubig=np.zeros([len(u_n.vector().get_local()),num_steps])
	# Time-stepping
	u = Function(V)
	t = 0
	for n in range(num_steps):
		
		# Update current time
		t += dt
		f1.u_n=u_n
		f2.u_n=u_n
		u_D.t=t
		 
		solve(a == l, u,bc, solver_parameters={'linear_solver':'gmres','preconditioner':'ilu'})
		u_n.assign(u)
		ubig[:,n]=u_n.vector().get_local()
		# Plot solution
		#plot(u)
		if save_sim:
			if n % 10 == 0:
				vtkfile<< (u,t)
		maxu=np.max(u.compute_vertex_values(mesh))
		maxuvect.append(maxu)
		if maxu>uig:
			maxu=1000
			break

		# Update previous solution
		u_n.assign(u)
	if plotmaxuvect:
		fig,ax=plt.subplots()
		tvect=np.linspace(0,t*L**2/alpha,n+2)
		ax.plot(tvect,epsilon1*Ta*np.array(maxuvect)+Ta-273)
		ax.set_xlabel('Time (Years)')
		ax.set_ylabel('Maximum Temperature ($^o$C)')
		fig.savefig('maxtemps/maxuvect'+str(To)+str(phi)+str(L)+str(reactnum)+'.pdf',bbox_inches='tight')
	return [ubig,maxu,epsilon1*Ta*np.array(maxuvect)+Ta-273]

def criticalL(X,dim=1,phi=0,To=0,reactnum=0):
	#dim =3
	logL1=-2
	logL2=2
	output=10**logL2
	[u,umax,maxuvect]=fenicsSolver(X,dimension=dim,L=10**logL2,phi=phi,To=To,reactnum=reactnum)
	if umax<1000:
		return output
	[u,umax,maxuvect]=fenicsSolver(X,L=10**logL1,dimension=dim,phi=phi,To=To,reactnum=reactnum)
	if umax>=1000:
		return 10**logL1
	
	logLc=(logL2+logL1)/2
	for i in range(20):
		[u,umax,maxuvect]=fenicsSolver(X,L=10**logLc,dimension=dim,phi=phi,To=To,reactnum=reactnum)
		if umax>=1000:
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
	thetabars=[np.mean(p1[0:N]),np.mean(p2[0:N]),np.mean(p3[0:N]),np.mean(p4[0:N])]
	Vars=[np.var(p1[0:N],ddof=1),np.var(p2[:N],ddof=1),np.var(p3[0:N],ddof=1),np.var(p4[0:N],ddof=1)]
	B=N*np.var(thetabars,ddof=1)
	W=np.mean(Vars)
	Var=(N-1)/N*W+B/N
	rhat=np.sqrt(Var/W)
	output['sample']=sample
	output['rhat']=rhat
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

def determineCriticalLs(data_run):
	data=assemble(data_run)
	results=generateData(data)
	X=transformArray(results)
	Lcr=Parallel(n_jobs=4)(delayed(criticalL)(x) for x in X)

	fig4,ax4=plt.subplots()
	density=scist.gaussian_kde(Lcr)
	n,y,_ =ax4.hist(Lcr,bins=50,density=True)
	ax4.plot(y,density(y))
	fig4.savefig('Single_output/hist_Lcr.pdf')
	return Lcr


def phicriticals(X):
	phis=np.linspace(0,2*pi)
	To=8
	Lcr=Parallel(n_jobs=4)(delayed(criticalL)(X,phi=phi,To=To) for phi in phis)
	
	fig4,ax4=plt.subplots()
	ax4.plot(phis,Lcr)
	ax4.set_xlabel('\phi')
	ax4.set_ylabel('$L_{cr}$')
	fig4.savefig('Periodic_BC/phi_critical.pdf')	
	return Lcr

def tocriticals(X):
	Tos=np.linspace(0,20)
	phi=0
	Lto=Parallel(n_jobs=4)(delayed(criticalL)(X,phi=phi,To=To) for To in Tos)
	
	fig4,ax4=plt.subplots()
	ax4.plot(Tos,Lto)
	ax4.set_xlabel('$T_o$')
	ax4.set_ylabel('$L_{cr}$')
	fig4.savefig('Periodic_BC/tos_critical.pdf')	
	return Lto

def ignitionTime(X,phi=0,To=0,L=5,dim=1):
	sol=fenicsSolver(X,L=L,phi=phi,To=To,dimension=dim)
	tig=((len(sol[2]))-1)/1000
	return tig

def multidimTig(X):
	Tig=[]
	dims=[1,2,3]
	Ls=[5.2,8,8]
	for i in range(3):
		Tig.append(ignitionTime(X,dim=dims[i],L=Ls[i]))
	return Tig+list(X)

def determineTigs(data_run,num_cores=4,samplesize=500):
	data=ld.assemble(data_run)
	burned=ld.removeBurn(data,10000)
	resultsburn=ld.generateData(burned,burnin=True)
	X=transformArray(resultsburn)
	Tig=Parallel(n_jobs=num_cores)(delayed(multidimTig)(X[random.randint(0,len(X))]) for i in range(samplesize))
	Tdf=pd.DataFrame(Tig)
	Tdf.to_csv('Tigs.csv')
	return Tdf


if __name__=='__main__':
	#determineCriticalLs('SIM')
	#testCases()
	"""
	L=4.128
	To=8
	phi=0
	medianvals=[4.395,4.7666,3.1095,8.7515,5.0908,3.842]
	#Lcr=criticalL(medianvals,To=To,phi=phi)
	L=3.2
	sol0=fenicsSolver(medianvals,L=L,To=To,phi=phi,plotmaxuvect=True)
	phi=pi/2
	sol1=fenicsSolver(medianvals,L=L,To=To,phi=phi,plotmaxuvect=True)
	phi=pi
	sol2=fenicsSolver(medianvals,L=L,To=To,phi=phi,plotmaxuvect=True)
	phi=3*pi/2
	sol3=fenicsSolver(medianvals,L=L,To=To,phi=phi,plotmaxuvect=True)
	#sol1=fenicsSolver(medianvals,L=L,To=To,save_sim=True,plotmaxuvect=True,reactnum=1)
	#sol2=fenicsSolver(medianvals,L=L,To=To,save_sim=True,plotmaxuvect=True,reactnum=2)
	#Lphi=phicriticals(medianvals)
	"""
	Tigs=determineTigs('EXP_Q')
	
