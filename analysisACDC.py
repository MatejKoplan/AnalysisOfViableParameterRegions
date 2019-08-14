from solver import Solver, Region
from numpy import random
import pickle
import os.path as path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
from acdc import ACDC    
from deap import creator, base, tools, algorithms   

parameter_values = {  "transcription": {"min": 0.01, "max": 50}, 
			"translation": {"min": 0.01, "max": 50},     
			"rna_degradation": {"min": 0.1, "max": 100},      
			"protein_degradation": {"min": 0.001, "max": 50},        
			"hill": {"min": 0.1, "max": 5},      
			"Kd": {"min": 0.01, "max": 250},               
			}  

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)		
toolbox = base.Toolbox()	 
toolbox.register("candidate", Solver.generateCandidate) 

model = ACDC(parameter_values, np.array(["transcription", "hill", "translation", "rna_degradation", "protein_degradation", "Kd", "Kd", "Kd", "Kd"]), np.array([0, 37, 0, 280, 0, 280]))
solver = Solver(model)     

folder = path.join(".", "ACDC", "exper0")	 	
				
file1 =  path.join(folder, "acdc_Region00ViableSet_Iter10.p")              
file2 =  path.join(folder, "acdc_Region01ViableSet_Iter10.p")  

viablePointsA = pickle.load(open(file1, "rb"))  
viablePointsB = pickle.load(open(file2, "rb"))  

regionA = Region(viablePointsA, model, "regionA") 
regionA.fitPCA()
regionB = Region(viablePointsB, model, "regionB") 
regionB.fitPCA() 

number = np.size(viablePointsA, 0) 
rndPoint = np.random.randint(number, size=1) 

#plot bistable dynamics
plt.rcParams.update({'font.size': 12})  
ax1=plt.subplot(1, 2, 1)    
plota = [5.91, 4.16, 47.35, 0.63, 1.12, 165.03, 111.2, 101.5, 0.48]
model.plotModel(plota, show=False, xlabel=r"Time [$h$]" + "\n \n" + r"$\mathbf{(a)}$")    

number = np.size(viablePointsB, 0)   
rndPoint = np.random.randint(number, size=1)   

#plot oscillatory dynamics 
ax1=plt.subplot(1, 2, 2)     
plotb = [26.26, 3.82, 18.14, 0.92, 1.29, 151.42, 197.61, 41.88, 11.7]
model.plotModel(plotb, show=False, xlabel=r"Time [$h$]" + "\n \n" + r"$\mathbf{(b)}$")     
plt.show()     


#sample few random points from viable regions for region 2     
readFromFile = False                

###	                                       ### 
###  SSA simultions for bistable dynamics  ### 
###	                                       ###       
numSamples = 6
if readFromFile:
	sampleNumbers2 = pickle.load(open("stochastic_samples_numbers_acdc_a", "rb"))
else:
	num2 = viablePointsA.shape[0]  
	sampleNumbers2 = np.random.choice(num2, numSamples, replace=False)   
	pickle.dump(sampleNumbers2, open("stochastic_samples_numbers_acdc_a", "wb+"))    
	
#plot few simulations (ode + ssa)
t = np.linspace(0, model.T, model.N)
if readFromFile:
	samples = pickle.load(open("stochastic_samples_acdc_a","rb")) 	
else:	
	samples = viablePointsA[sampleNumbers2,:]
	pickle.dump(samples, open("stochastic_samples_acdc_a", "wb+"))   

samplesA = samples 	
	
ts_total_stochastic = [] 
Y_total_stochastic = []
t_total_ode = []
Y_total_ode = []

if readFromFile:
	t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic = pickle.load(open("stochastic_simulations_acdc_a", "rb"))  
for i in range(numSamples): 
	print(i)  
	ax1=plt.subplot(4, 3, i + numSamples + 1) 
	ax1.set_title("Sample " + str(i + numSamples + 1), fontsize=12) 
	sample = samples[i]  
	if readFromFile:
		ts = ts_total_stochastic[i]   
		Y = Y_total_stochastic[i]  
		Y_ode = Y_total_ode[i]  
		t = t_total_ode[i]  
	else:
		ts, Y = model.ACDCStochastic(sample)
		Y = Y/model.omega 
		Y_ode = model.simulate(sample)   
		ts_total_stochastic.append(ts) 
		Y_total_stochastic.append(Y)  
		t_total_ode.append(t)   
		Y_total_ode.append(Y_ode) 
	
	lines = plt.plot(t, Y_ode[:,1], t, Y_ode[:,3], ts, Y[:,1], ts, Y[:,3])   
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#15A357")  
	plt.setp(lines[1], linestyle = "--", linewidth=1.5, alpha = 1, c="#A62B21")   
	plt.setp(lines[2], linewidth=1.5, alpha= 0.65, c="#15A357")  
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#A62B21")  

		
	if i + numSamples == 9 or i + numSamples == 10 or i + numSamples == 11:
		plt.xlabel(r"Time [$h$]")    
	if i + numSamples == 0 or i + numSamples == 3 or i + numSamples == 6 or i + numSamples == 9:
		plt.ylabel('Concentration [$nM$]')    	

if not readFromFile:
	pickle.dump((t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic) ,open("stochastic_simulations_acdc_a", "wb+"))       	


###	                                           ###
###  SSA simulations for oscillatory dynamics  ###
###	                                           ###  
numSamples = 6
if readFromFile:
	sampleNumbers2 = pickle.load(open("stochastic_samples_numbers_acdc_b", "rb"))
else:
	num2 = viablePointsB.shape[0]  
	sampleNumbers2 = np.random.choice(num2, numSamples, replace=False)   
	pickle.dump(sampleNumbers2, open("stochastic_samples_numbers_acdc_b", "wb+"))    
	
#plot few simulations (ode + ssa)
t = np.linspace(0, model.T, model.N)
if readFromFile:
	samples = pickle.load(open("stochastic_samples_acdc_b","rb")) 	
else:	
	samples = viablePointsB[sampleNumbers2,:]
	pickle.dump(samples, open("stochastic_samples_acdc_b", "wb+"))   

samplesB = samples     	
	
ts_total_stochastic = [] 
Y_total_stochastic = []
t_total_ode = []
Y_total_ode = []

if readFromFile:
	t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic = pickle.load(open("stochastic_simulations_acdc_b", "rb"))  
for i in range(numSamples): 
	print(i)   
	ax1=plt.subplot(4, 3, i	+ 1) 
	ax1.set_title("Sample " + str(i + 1), fontsize=12)
	sample = samples[i]  
	if readFromFile:
		ts = ts_total_stochastic[i]   
		Y = Y_total_stochastic[i]  
		Y_ode = Y_total_ode[i]  
		t = t_total_ode[i]  
	else:
		ts, Y = model.ACDCStochastic(sample)
		Y = Y/model.omega 
		Y_ode = model.simulate(sample)   
		ts_total_stochastic.append(ts) 
		Y_total_stochastic.append(Y)  
		t_total_ode.append(t)   
		Y_total_ode.append(Y_ode) 
	
	lines = plt.plot(t, Y_ode[:,1], t, Y_ode[:,3], ts, Y[:,1], ts, Y[:,3])   
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha= 1, c="#15A357" )  
	plt.setp(lines[1], linestyle = "--", linewidth=1.5, alpha = 1, c="#A62B21")   
	plt.setp(lines[2], linewidth=1.5, alpha= 0.65, c="#15A357")  
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#A62B21")  

		
	if i == 9 or i == 10 or i == 11:
		plt.xlabel(r"Time [$h$]")    
	if i == 0 or i == 3 or i == 6 or i == 9:
		plt.ylabel('Concentration [$nM$]')    	
plt.show()

if not readFromFile:
	pickle.dump((t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic) ,open("stochastic_simulations_acdc_b", "wb+")) 	
	
#plot boxplots
repetitions = 100      
amps = [] 
pers = []       
ode_amps = [] 
ode_pers = [] 
margins = []
ode_margins = [] 

if readFromFile:
	pers, amps = pickle.load(open("stochastic_persamps_acdc_b","rb"))  
for i in range(numSamples): 
	print(i) 
	sample = samples[i] 
	per, amp = model.getPerAmp(sample, mode="ode", indx=1)   
	ode_amps.append(amp)   
	ode_pers.append(per)    
	
	if not readFromFile:	
		curra = []
		currp = []
		for j in range(repetitions):
			per, amp = model.getPerAmp(sample, mode="ssa", indx=1)  
			amp = amp/model.omega    
			curra.append(amp)  
			currp.append(per)    		
		amps.append(curra)  
		pers.append(currp) 

if not readFromFile:	
	pickle.dump((pers, amps), open("stochastic_persamps_acdc_b", "wb+"))  

if readFromFile:
	margins = pickle.load(open("stochastic_margins_acdc", "rb"))  
for i in range(numSamples):
	print(i) 
	sample = samplesA[i] 
	margin = model.getMargin(sample, mode="ode", indx1=3, indx2=1) 
	ode_margins.append(margin) 
	if not readFromFile:
		currm = []
		for j in range(repetitions):
			margin = model.getMargin(sample, mode="ssa", indx1=3, indx2=1)  
			currm.append(margin)
		margins.append(currm)

if not readFromFile:
	pickle.dump((margins), open("stochastic_margins_acdc", "wb+")) 
		
plt.subplot(1,3,1)	
bp = plt.boxplot(amps)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_amps[i], ode_amps[i]], linewidth=3, alpha=1, color="black")
	
plt.ylabel('Amplitude [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(a)}$')      
print(ode_amps)   


plt.subplot(1,3,2)
bp = plt.boxplot(pers)   
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_pers[i], ode_pers[i]], linewidth=3, alpha=1, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample \n \n $\mathbf{(b)}$')      
print(ode_pers)   	

plt.subplot(1,3,3) 
bp = plt.boxplot(margins)   
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_margins[i], ode_margins[i]], linewidth=3, alpha=1, color="black")
	
plt.ylabel('$|\overline{X} - \overline{Y}|$ [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(c)}$')      
print(ode_margins)    	
plt.show()  


solver_object = Solver(model)    

filesA1 = ["acdc_Region00ViableSet_Iter1.p", "acdc_Region00ViableSet_Iter9.p"]    
filesA2 = ["acdc_Region00CandidateSet_Iter2.p", "acdc_Region00CandidateSet_Iter10.p"] 
filesB1 = ["acdc_Region01ViableSet_Iter1.p", "acdc_Region01ViableSet_Iter9.p"]    
filesB2 = ["acdc_Region01CandidateSet_Iter2.p", "acdc_Region01CandidateSet_Iter10.p"]    

setGA = Region(pickle.load(open(path.join(".", "ACDC", "exper0", "acdcViableSet_IterGA.p"), "rb")), model, "regionGA")
setGA.fitPCA()  
setGA.updateIter()    
fpca = setGA.pca      

candidateSet = random.multivariate_normal([0]*setGA.model.nParams, np.diag(setGA.pca.explained_variance_)*setGA.varScale, solver_object.nsamples)				
candidateSet = setGA.inverse_transform(candidateSet)  

#check if parameter values are not out of range  		
inBounds = list() 
for cand in candidateSet: 				
	if not solver_object.checkOutAllBounds(cand):   
		inBounds.append(cand)  
inBounds = np.array(inBounds)   		
candidateSet = inBounds 

X = fpca.transform(setGA.points)   
Y = fpca.transform(candidateSet) 

regionAPCA = fpca.transform(viablePointsA)
regionBPCA = fpca.transform(viablePointsB)  
				
#plot viable regions in PCA space  
ax1=plt.subplot(1, 3, 1)
ax1.title.set_text("Iteration 0")       

plt.scatter(Y[:, 0], Y[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)  
plt.scatter(X[:, 0], X[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)   		

plt.xlabel('PC 1')      
plt.ylabel('PC 2')  

i = 2
iters = [1, 10] 
for filea1, filea2, fileb1, fileb2  in zip(filesA1, filesA2, filesB1, filesB2):  	 

	filea1 =  path.join(folder, filea1)      
	filea2 =  path.join(folder, filea2)      
	fileb1 =  path.join(folder, fileb1)      
	fileb2 =  path.join(folder, fileb2)        
			
	viablePoints1 = pickle.load(open(filea1, "rb"))     
	candidatePoints1 = pickle.load(open(filea2, "rb"))     
	viablePoints2 = pickle.load(open(fileb1, "rb"))     
	candidatePoints2 = pickle.load(open(fileb2, "rb")) 

	region1PCA = fpca.transform(viablePoints1) 	
	region1PCAcandidates = fpca.transform(candidatePoints1) 
	region2PCA = fpca.transform(viablePoints2) 	
	region2PCAcandidates = fpca.transform(candidatePoints2) 	

	ax1=plt.subplot(1, 3, i)
	ax1.title.set_text("Iteration " + str(iters[i-2]))

	plt.scatter(region1PCAcandidates[:, 0], region1PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True) 
	plt.scatter(region2PCAcandidates[:, 0], region2PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)     
	plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)
	plt.scatter(region2PCA[:, 0], region2PCA[:, 1], c="#15A357", alpha=1, edgecolor="#158257", rasterized=True)     
	plt.xlabel('PC 1')     
	i += 1 
plt.show() 

viableSets = [regionA, regionB]    		
paramNames = [r"$\alpha$", r"$n$",r"$\beta$", r"$\delta_m$",r"$\delta_p$", r"$Kd_a$", r"$Kd_b$", r"$Kd_c$", r"$Kd_d$"] 
units = [r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$nM$]", r"[$nM$]", r"[$nM$]", r"[$nM$]"]     
solver_object.plotParameterVariances(viableSets, names=paramNames, units=units)   


#calculate viable volumes  
vol1 = solver_object.getViableVolume([regionA], 100000)      
vol2 = solver_object.getViableVolume([regionB], 100000)   
print("Volume 2 is " + str(vol2/vol1) + "greater than volume 1")  
















