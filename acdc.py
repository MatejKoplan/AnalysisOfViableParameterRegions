import numpy as np 
import math
import peakutils
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint 
from repressilator import Repressilator
import scipy.signal as signal

	
'''
The deterministic model of biological ACDC  TOGGLE SWITCH + REPRESSILATOR
'''
class ACDC: 
	
	def __init__(self, parameter_values, params, initial_conditions, dt = 0.001): 
		self.nParams = len(params)   
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 	
		self.dt = dt
		self.T = 48 #hours
		self.N = int(self.T/self.dt)   
		self.ts = np.linspace(0, self.T, self.N) 
		self.amp = 200 #[nM]  				  	
		self.sample_rate 		= 0.0033333333 #[Hz]   
		self.samples_per_hour 	= (1/self.dt)		
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600))    	         
		self.per = self.T/8   		
		self.ideal = self.amp*(np.sin(math.pi*(self.ts)/self.per - math.pi/2) + 1)  				
		self.idealBistable = [2*self.amp,0]      		
		self.nS = self.N/self.jump  
		self.dF = self.sample_rate/self.nS  
		self.idealF = self.getFrequencies(self.ideal) 
		self.thresholdOscillations = -2250 #harmonics do not deviate 15nM                  
		self.thresholdBistability = -9000 #signal does not deviate more than 4nM approx                                
		self.modes = [self.evalBistability, self.evalOscillations]      
		self.omega = 1#nm^-1

	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y)) 
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res 
				
	def isViable(self, point): 
		fitnessBistability = self.evalBistability(point) 
		fitnessOscillations = self.evalOscillations(point)   			
		return fitnessBistability[0] > self.thresholdBistability or fitnessOscillations[0] > self.thresholdOscillations	
	
	def evalBistability(self, candidate):
		Y = np.array(self.simulate(candidate, [0, 1.5*self.amp, 0, 0.5*self.amp, 0, 0])) 
		startFrom = int((self.N*10.0)/self.T)
		x1 = Y[:,1] 	
		x2 = Y[:,3]	 	
		x1 = x1[startFrom::self.jump]  
		x2 = x2[startFrom::self.jump] 		
		c1 = x1 - self.idealBistable[0]  
		c2 = x2 - self.idealBistable[1] 
		c1 = np.dot(c1,c1) 
		c2 = np.dot(c2,c2)  
		cost1 = c1 + c2 
				
		Y = np.array(self.simulate(candidate, [0, 0.5*self.amp, 0, 1.5*self.amp, 0, 0]))     
		x1 = Y[:,1] 	 
		x2 = Y[:,3]		
		x1 = x1[startFrom::self.jump]    
		x2 = x2[startFrom::self.jump]  		
		c1 = x1 - self.idealBistable[1] 
		c2 = x2 - self.idealBistable[0]
		c1 = np.dot(c1,c1) 
		c2 = np.dot(c2,c2)  
		cost2 = c1 + c2 		
		cost = -(cost1 + cost2) 
		return cost, 
		
	
	def evalOscillations(self, candidate): 
		Y = np.array(self.simulate(candidate)) 
		
		x = Y[:,1]		
		fftData = self.getFrequencies(x)    
		#take only first 10 harmonics  
		fftData = fftData[0:10] 
		idealF = self.idealF[0:10] 
		diff = fftData - idealF   
		cost = -np.dot(diff, diff)  
		return cost, 					
	
	#simulates a candidate
	def simulate(self, candidate, y0 = None):  
		if y0 == None: 
			y0 = self.y0 	
		return odeint(self.acdcModelOde, y0, self.ts, args=(candidate,))  				
		
	def plotModel(self, subject, show=True, xlabel = None, mode="ode"):  
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N)
			Y = self.simulate(subject)  		
		else:
			ts, Y = self.ACDCStochastic(subject)  
					
		Y = np.array(Y) 

		p1 = Y[:,1]  
		p3 = Y[:,3]      
		p5 = Y[:,5] 
		
		lines = plt.plot( ts, p1, ts, p3, ts, p5)  
		plt.setp(lines[0], linewidth=2, c="#15A357")
		plt.setp(lines[1], linewidth=2, c="#A62B21")
		plt.setp(lines[2], linewidth=2, c="#0E74C8")      		
		plt.ylabel('Concentration [$nM$]')     
		if xlabel == None:
			plt.xlabel("Time [$h$]")    
		else:
			plt.xlabel(xlabel)  
		plt.legend(('$X$', '$Y$', '$Z$',), loc='upper right')   
		if show: 	
			plt.show() 
			
			

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol 

	def acdcModelOde(self, Y, t, can):  		   
		alpha = can[0]
		n = can[1]
		beta = can[2]
		deltaRNA = can[3] 
		deltaP = can[4]
		kda = can[5]  
		kdb = can[6]
		kdc = can[7]
		kde = can[8]  
		mx = Y.item(0)
		my = Y.item(2)  
		mz = Y.item(4)
		x = Y.item(1)
		y = Y.item(3) 
		z = Y.item(5)   
		
		#in case of math range error
		try: 
			dmx = -deltaRNA*mx + alpha/(1 + math.pow(z/kda, n) + math.pow(y/kdb, n))    
			dmy = -deltaRNA*my + alpha/(1 + math.pow(x/kdc, n))       
			dmz = -deltaRNA*mz + alpha/(1 + math.pow(y/kde, n))      
		except (OverflowError, ValueError):
			dmx = -deltaRNA*mx + alpha    
			dmy = -deltaRNA*my + alpha   
			dmz = -deltaRNA*mz + alpha    
			
		dpx = beta*mx - deltaP*x
		dpy = beta*my - deltaP*y 
		dpz = beta*mz - deltaP*z
		
		return np.array([dmx, dpx, dmy, dpy, dmz, dpz])  

	def getPerAmp(self, subject, mode="ode", indx=0): 
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N) 
			Y = self.simulate(subject)    				
		else:
			ts,Y = self.ACDCStochastic(subject) 
		ts = np.array(ts) 
		Y = np.array(Y) 
		sig = Y[:, indx]
		indx_max, properties = signal.find_peaks(sig, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)      
		indx_min, properties = signal.find_peaks(sig*-1, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)     

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
		
		print("amp" + str(amp)) 
		print("per" + str(per))    	
		
		return per, amp 
	

	def getMargin(self, subject, mode="ode", indx1=3, indx2=1):
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N) 
			Y = self.simulate(subject)  
		else:
			ts,Y = self.ACDCStochastic(subject)
			
		ts = np.array(ts) 
		Y = np.array(Y) 			
		
		Y = Y[ts>=10,:]	
		
		margin = np.absolute(np.mean(Y[:, indx1]) - np.mean(Y[:, indx2])) 
		print(margin) 
		
		return margin
		
		
	def ACDCStochastic(self, can):
		omega = self.omega 
		y_conc = np.array(self.y0*omega).astype(int) 

		Y_total = []
		Y_total.append(y_conc)
		t = 0 
		T = [] 
		T.append(t)
		
		#get kinetic rates 
		alpha = can[0]
		n = can[1]
		beta = can[2]
		deltaRNA = can[3] 
		deltaP = can[4]
		kda = can[5]  
		kdb = can[6]
		kdc = can[7]
		kde = can[8] 
		
		N = np.zeros((6,12)) #6 species, 12 reactions

		N[0,0] = -1
		N[0,1] = 1
		
		N[1,6] = 1
		N[1,7] = -1
		
		N[2,2] = -1
		N[2,3] = 1

		N[3,8] = 1
		N[3,9] = -1
		
		N[4,4] = -1
		N[4,5] = 1

		N[5,10] = 1
		N[5,11] = -1
		
		while t < self.T:
			#choose two random numbers 
			r = np.random.uniform(size=2)
			r1 = r[0]
			r2 = r[1] 
						
			#get propensities
			a = np.zeros(12)
			a[0] = deltaRNA*y_conc[0] 
			a[1] = alpha/(1.0 + math.pow(y_conc[5]/(kda*omega), n) + math.pow(y_conc[3]/(kdb*omega), n))*omega
			
			a[2] = deltaRNA*y_conc[2] 
			a[3] = alpha/(1.0 + math.pow(y_conc[1]/(kdc*omega), n))*omega 
			
			a[4] = deltaRNA*y_conc[4] 
			a[5] = alpha/(1.0 + math.pow(y_conc[3]/(kde*omega), n))*omega   
			
			a[6] = beta*y_conc[0]
			a[7] = deltaP*y_conc[1]

			a[8] = beta*y_conc[2]
			a[9] = deltaP*y_conc[3]

			a[10] = beta*y_conc[4] 
			a[11] = deltaP*y_conc[5] 
			
			asum = np.cumsum(a)
			a0 = np.sum(a)  
			#get tau
			tau = (1.0/a0)*np.log(1.0/r1)      
		
			#select reaction 
			reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element			
		
			#update concentrations
			y_conc = y_conc + N[:,reaction_number]   	
			Y_total.append(y_conc) 
			#update time
			t = t + tau  
			T.append(t)
		
		T = np.array(T)
		Y_total = np.array(Y_total) 
		return T, Y_total	
	
	

