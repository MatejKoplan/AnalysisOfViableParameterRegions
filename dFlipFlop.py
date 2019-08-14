import numpy as np 
import math
from math import pow
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint  
import scipy.signal as signal   
 	
'''
The deterministic model of biological D flip-flop in master slave cinfiguration
'''
class DFlipFlop: 
	
	def __init__(self, parameter_values, params, initial_conditions, threshold = -540000*1e3, dt = 0.01, omega1 = 0, omega2 = 0): #TO DO: popravi parameter viable points 
		#dictionary of form parameter: {min: val, max: val, ref: val}   
		self.nParams = len(params)  
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 
		self.threshold = threshold 	
		self.dt = dt
		self.T = 96 #in hours 
		self.N = int(self.T/self.dt) 
		self.ts = np.linspace(0, self.T, self.N)  
		self.per = self.T/4 
		self.amp = 100 #[nM]
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)
		self.jump = int(self.samples_per_hour/(self.sample_rate*3600))
		#clk signal
		self.CLK = [self.getClock(x) for x in np.linspace(0, self.T, self.N)]   
		#ideal response 
		self.nS = self.N/self.jump 
		self.ideal = [0]*self.N  
		self.ideal[0:int(self.N /4)] = [self.amp]*int(self.N /4) 
		self.ideal[2*int(self.N /4):3*int(self.N/4)] = [self.amp]*int(self.N /4)    						 		
		self.idealF = self.getFrequencies(self.ideal) 	
		self.threshold = -10*100  #10nM -+ from ideal signal harmonics, only first 10 are selected 
		self.modes = [self.eval] 
		self.omega1 = omega1    
		self.omega2 = omega2   
		self.dil = 0.6 #to do (konstanta) 0.01/min = 0.6/h  
		self.omega = 1  
			
	def getClock(self, t):
		return self.amp*(np.sin(2*math.pi*(t)/self.per) + 1)/2

	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res
		
	#evaluates a candidate 
	def eval(self, candidate):
		Y = np.array(self.simulate(candidate)) 		
		p1 = Y[:,2] #take q    
		fftData = self.getFrequencies(p1)   
		#take only first 10 harmonics  
		fftData = fftData[0:10] 
		idealF = self.idealF[0:10] 
		diff = fftData - idealF   
		cost = -np.dot(diff, diff)      		     		
		return cost,          
		
	def isViable(self, point): 	
		fitness = self.eval(point) 			
		return fitness[0] >= self.threshold  

	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res 
		
	#simulates a candidate
	def simulate(self, candidate):		
		return odeint(self.flipFlopModelOde, self.y0, self.ts, args=(candidate,))  			 

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol

	def plotModel(self, subject, mode="ode"):
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N)
			Y = self.simulate(subject) 			
			Y = np.array(Y) 
			p1 = self.CLK   
		else:
			omega = 1.2044 
			ts, Y = self.flipflopStochastic(subject) 	
			Y = np.array(Y)
			p1 = Y[:,4] 
		  	
		p2 = Y[:,0]     		
		p3 = Y[:,1]   
		p4 = Y[:,2]   
		p5 = Y[:,3]   
    
		lines = plt.plot(ts, p1, '#15A357', ts, p2, 'k', ts, p3, 'k--', ts, p4, '#A62B21', ts, p5, '#0E74C8')  
		plt.setp(lines[0], linewidth=2)
		plt.setp(lines[1], linewidth=2) 
		plt.setp(lines[2], linewidth=2)  
		plt.setp(lines[3], linewidth=2)  
		plt.setp(lines[4], linewidth=2)  
		
		plt.ylabel(r'Concentration [$nM$]') 
		plt.xlabel(r"Time [$h$]")   
		plt.legend(('$CLK$', '$a$', '$a_{c}$', '$q$', '$qc$'), loc='upper right')   		
		plt.show() 				
		
	def getPerAmp(self, subject, mode="ode", indx1=2, indx2=3):  
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N) 
			Y = self.simulate(subject)    				
		else:
			ts,Y = self.flipflopStochastic(subject) 
		ts = np.array(ts) 
		Y = np.array(Y) 
		sig = Y[:, indx1] - Y[:, indx2] 
		indx_max, properties = signal.find_peaks(sig, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/4)              
		indx_min, properties = signal.find_peaks(sig*-1, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/4)                   

		amps = [] 
		pers = []   
		for i in range(min(len(indx_max), len(indx_min))):
			amps.append((sig[indx_max[i]] - sig[indx_min[i]])/2) 			
			if i + 1 < len(indx_max):
				pers.append(ts[indx_max[i + 1]] - ts[indx_max[i]])
			if i + 1 < len(indx_min):
				pers.append(ts[indx_min[i + 1]] - ts[indx_min[i]])
		
		if len(amps) > 0 and len(pers) > 0:
			amps = np.array(amps)   	
			pers = np.array(pers)  
			
			amp = np.mean(amps)	
			per = np.mean(pers) 
		else:
			amp = 0
			per = 0  
		
		print("amp" + str(amp/2.0))  
		print("per" + str(per))  
		return per, amp/2.0   
		
	def flipFlopModelOde(self, Y, t, can):  
		a     = Y.item(0) 
		not_a = Y.item(1)
		q     = Y.item(2)
		not_q = Y.item(3)
		d = not_q
		sumP = a + not_a + q + not_q
				
		alpha1 	= can[0]
		alpha2 	= can[1]
		alpha3 	= can[2]
		alpha4 	= can[3]
		delta1 	= can[4]
		delta2 	= can[5]
		Kd 		= can[6] 
		n 		= can[7]

		if self.omega2 == 0:
			Km 		= can[8] #dissociation constant for enzyme-protein complex  
			E 		= can[9]  
		else:
			Km = 0
			E = 0 
		
		clk = self.getClock(t) 
		
		
		try:
			da_dt     = alpha1*(pow(d/Kd, n)/(1 + pow(d/Kd, n) + pow(clk/Kd, n) + self.omega1*pow(d/Kd, n)*pow(clk/Kd, n))) + alpha2*(1/(1 + pow(not_a/Kd, n))) - (self.omega2*delta1 + (1 - self.omega2)*((E*delta1)/(Km + sumP) + self.dil))*a #self.dil = dilution rate     	
			dnot_a_dt = alpha1*(1/(1 + pow(d/Kd, n) + pow(clk/Kd, n) + self.omega1*pow(d/Kd, n)*pow(clk/Kd, n))) + alpha2*(1/(1 + pow(a/Kd, n))) - (self.omega2*delta1 + (1 - self.omega2)*((E*delta1)/(Km + sumP)+ self.dil))*not_a   
			dq_dt     = alpha3*((pow(a/Kd, n)*pow(clk/Kd, n))/(1 + pow(a/Kd, n) + pow(clk/Kd, n) + pow(a/Kd, n)*pow(clk/Kd, n))) + alpha4*(1/(1 + pow(not_q/Kd, n))) - (self.omega2*delta2 + (1 - self.omega2)*((E*delta2)/(Km + sumP) + self.dil))*q  
			dnot_q_dt = alpha3*((pow(not_a/Kd, n)*pow(clk/Kd, n))/(1 + pow(not_a/Kd, n) + pow(clk/Kd, n) + pow(not_a/Kd, n)*pow(clk/Kd, n))) + alpha4*(1/(1 + pow(q/Kd, n))) - (self.omega2*delta2 + (1 - self.omega2)*((E*delta2)/(Km + sumP)+ self.dil))*not_q   
		except (OverflowError, ValueError): 
			da_dt = 0
			dnot_a_dt = 0
			dq_dt = 0
			dnot_q_dt = 0 
			#print("OverflowError") 
		return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt]) 
	
	
	def flipflopStochastic(self, can): 
		omega = self.omega #nm^-1
		omega1 = self.omega1 
		omega2 = self.omega2    
		
		y_conc = np.array(self.y0*omega).astype(int) 
		clk = int(self.getClock(0)*omega) 
		y_conc = np.append(y_conc, clk) 
		
		Y_total = []
		Y_total.append(y_conc)
		t = 0 
		T = [] 
		T.append(t)
	
		#get kinetic rates 
		alpha1 	= can[0]
		alpha2 	= can[1]
		alpha3 	= can[2]
		alpha4 	= can[3]
		delta1 	= can[4]
		delta2 	= can[5]
		Kd 		= can[6]
		n 		= can[7] 
		
		if self.omega2 == 0:
			Km 		= can[8] #dissociation constant for enzyme-protein complex  
			E 		= can[9]  
		else:
			Km = 0
			E = 0 
	
		N = np.zeros((5,12)) #4 species + clk, 12 reactions 
		
		N[0,0] = 1   
		N[0,1] = 1  
		N[0,2] = -1   
		N[1,3] = 1  
		N[1,4] = 1  
		N[1,5] = -1  
		N[2,6] = 1 
		N[2,7] = 1 
		N[2,8] = -1 
		N[3,9] = 1 
		N[3,10] = 1  
		N[3,11] = -1   		

		while t < self.T:
			#choose two random numbers 
			r = np.random.uniform(size=2)
			r1 = r[0]
			r2 = r[1]
			
			a     = y_conc[0]   
			not_a = y_conc[1] 
			q     = y_conc[2] 
			not_q = y_conc[3] 
			d = not_q  
			
			sumP = a + not_a + q + not_q    
			
			clk = int(self.getClock(t)*omega)        

			#get propensities
			p = np.zeros(12)
			p[0]  = alpha1*(pow(d/(Kd*omega), n)/(1 + pow(d/(Kd*omega), n) + pow(clk/(Kd*omega), n) + omega1*pow(d/(Kd*omega), n)*pow(clk/(Kd*omega), n)))*omega
			p[1]  = alpha2*(1/(1 + pow(not_a/(Kd*omega), n)))*omega
			p[2]  = (self.omega2*delta1 + (1 - self.omega2)*((E*delta1)/(Km + sumP) + self.dil))*a
			p[3]  = alpha1*(1/(1 + pow(d/(Kd*omega), n) + pow(clk/(Kd*omega), n) + omega1*pow(d/(Kd*omega), n)*pow(clk/(Kd*omega), n)))*omega 
			p[4]  = alpha2*(1/(1 + pow(a/(Kd*omega), n)))*omega
			p[5]  = (self.omega2*delta1 + (1 - self.omega2)*((E*delta1)/(Km + sumP)+ self.dil))*not_a 
			p[6]  = alpha3*((pow(a/(Kd*omega), n)*pow(clk/(Kd*omega), n))/(1 + pow(a/(Kd*omega), n) + pow(clk/(Kd*omega), n) + pow(a/(Kd*omega), n)*pow(clk/(Kd*omega), n)))*omega
			p[7]  = alpha4*(1/(1 + pow(not_q/(Kd*omega), n)))*omega
			p[8]  = (self.omega2*delta2 + (1 - self.omega2)*((E*delta2)/(Km + sumP) + self.dil))*q 
			p[9]  = alpha3*((pow(not_a/(Kd*omega), n)*pow(clk/(Kd*omega), n))/(1 + pow(not_a/(Kd*omega), n) + pow(clk/(Kd*omega), n) + pow(not_a/(Kd*omega), n)*pow(clk/(Kd*omega), n)))*omega
			p[10] = alpha4*(1/(1 + pow(q/(Kd*omega), n)))*omega 
			p[11] = (self.omega2*delta2 + (1 - self.omega2)*((E*delta2)/(Km + sumP)+ self.dil))*not_q    
			
			asum = np.cumsum(p) 
			p0 = np.sum(p)  
			#get tau
			tau = (1.0/p0)*np.log(1.0/r1)         
		
			#select reaction 
			reaction_number = np.argwhere(asum > r2*p0)[0,0] #get first element			
		
			#update concentrations
			y_conc = y_conc + N[:,reaction_number]
			y_conc[4] = clk 			
			Y_total.append(y_conc) 
			#update time
			t = t + tau  
			T.append(t)

		T = np.array(T)
		Y_total = np.array(Y_total)  
		return T, Y_total	
				
				

