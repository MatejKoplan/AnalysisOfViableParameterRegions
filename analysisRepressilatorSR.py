import solver
import pickle
import os.path as path
from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
from repressilator_selfrepressing import RepressilatorSR
from deap import creator, base, tools, algorithms  

plt.rcParams.update({'font.size': 12}) 

parameter_values = {  "transcription": {"min": 0.01, "max": 50},  
				"translation": {"min": 0.01, "max": 50}, 
				"protein_production": {"min": 0.1, "max": 50},         				
				"rna_degradation": {"min": 0.1, "max": 100},      
				"protein_degradation": {"min": 0.001, "max": 50},       
				"hill": {"min": 1, "max": 5},        
				"Kd": {"min": 0.01, "max": 250},
                                "ar_factor": {"min": 0.0, "max": 5},
                                "ar_Kd": {"min": 1, "max": 5}
				}  
				
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)		
toolbox = base.Toolbox()	 
toolbox.register("candidate", solver.Solver.generateCandidate) 
				
folder1 = path.join(".", "repressilatorSRCostOne", "exper0")      
folder2 = path.join(".", "repressilatorSRCostTwo", "exper0")      				
				
file1 =  path.join(folder1, "repres_Region0ViableSet_Iter10.p")      
file2 =  path.join(folder2, "repres_Region0ViableSet_Iter10.p")            

viablePoints1 = pickle.load(open(file1, "rb"))  
viablePoints2 = pickle.load(open(file2, "rb"))     

allViablePoints = np.vstack((viablePoints1, viablePoints2))
pca = PCA(n_components=2)
pca.fit(allViablePoints)

model1 = RepressilatorSR(parameter_values, np.array(["transcription", "transcription", "hill", "translation", "rna_degradation", "protein_degradation", "Kd", "ar_factor", "ar_Kd"]), np.array([0, 0, 10, 150, 0, 0]), mode=0) 
model2 = RepressilatorSR(parameter_values, np.array(["transcription", "transcription", "hill", "translation", "rna_degradation", "protein_degradation", "Kd", "ar_factor", "ar_Kd"]), np.array([0, 0, 10, 150, 0, 0]), mode=1) 

###                   ###
###  SSA simulations  ###
###                   ###  

#sample few random points from viable regions for region 2     
readFromFile = False                 

region2 = viablePoints2    
numSamples = 9
if readFromFile:
	sampleNumbers2 = pickle.load(open("stochastic_samples_numbers_repressilatorSR", "rb"))
else:
	num2 = region2.shape[0]  
	sampleNumbers2 = np.random.choice(num2, numSamples, replace=False)   
	pickle.dump(sampleNumbers2, open("stochastic_samples_numbers_repressilatorSR", "wb+"))    
	
#plot few simulations (ode + ssa)
t = np.linspace(0, model2.T, model2.N)
if readFromFile:
	samples = pickle.load(open("stochastic_samples_repressilatorSR","rb"))     
else:	
	samples = region2[sampleNumbers2,:]
	pickle.dump(samples, open("stochastic_samples_repressilatorSR", "wb+"))   
	
notWorking = samples[numSamples-3:numSamples,:] 
notWorking = np.copy(notWorking) 
notWorking[0,2] = 1 #low hill coefficient
notWorking[1,0] = 1 #low transcription 
notWorking[2,5] = 10 #high protein degradation   

allSamples = np.vstack((samples, notWorking)) 

ts_total_stochastic = [] 
Y_total_stochastic = []
t_total_ode = []
Y_total_ode = []

if readFromFile:
	t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic = pickle.load(open("stochastic_simulations_repressilatorSR", "rb"))  
for i in range(numSamples + 3): 
	print(i)  
	ax1=plt.subplot(4, 3, i + 1)

	ax1.set_title("Sample " + str(i + 1), fontsize=12)	 
	
	sample = allSamples[i] 
	if readFromFile:
		ts = ts_total_stochastic[i]   
		Y = Y_total_stochastic[i]  
		Y_ode = Y_total_ode[i]  
		t = t_total_ode[i]  
	else:
		ts, Y = model2.represilatorStochastic(sample)
		Y = Y/model2.omega 
		Y_ode = model2.simulate(sample)  
		ts_total_stochastic.append(ts)
		Y_total_stochastic.append(Y) 
		t_total_ode.append(t)  
		Y_total_ode.append(Y_ode) 
	
	lines = plt.plot(t, Y_ode[:,1], ts, Y[:,1])  
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#15A357")
	plt.setp(lines[1], linewidth=1.5, alpha = 0.65, c="#15A357") 


		
	if i == 9 or i == 10 or i == 11:
		plt.xlabel(r"Time [$h$]")    
	if i == 0 or i == 3 or i == 6 or i == 9:
		plt.ylabel('Concentration [$nM$]')    	
plt.show()


if not readFromFile:
	pickle.dump((t_total_ode, Y_total_ode, ts_total_stochastic, Y_total_stochastic) ,open("stochastic_simulations_repressilatorSR", "wb+"))       	
	
#plot boxplots
repetitions = 100    
amps = []
pers = []      
ode_amps = []
ode_pers = []

if readFromFile:
	pers, amps = pickle.load(open("stochastic_persamps_repressilatorSR","rb")) 
for i in range(numSamples): 
	print(i) 
	sample = samples[i]
	per, amp = model2.getPerAmp(sample, mode="ode", indx=1)   
	ode_amps.append(amp)   
	ode_pers.append(per)    
	
	if not readFromFile:	
		curra = []
		currp = []
		for j in range(repetitions):
			per, amp = model2.getPerAmp(sample, mode="ssa", indx=1)  
			amp = amp/model2.omega    
			curra.append(amp)  
			currp.append(per)   		
		amps.append(curra) 
		pers.append(currp) 

if not readFromFile:	
	pickle.dump((pers, amps), open("stochastic_persamps_repressilatorSR", "wb+"))  


plt.subplot(1,2,1)	
bp = plt.boxplot(amps)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_amps[i], ode_amps[i]], linewidth=3, color="black")
	
plt.ylabel('Amplitude [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(a)}$')  


print(ode_amps)   

plt.subplot(1,2,2) 	
bp = plt.boxplot(pers)   
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_pers[i], ode_pers[i]], linewidth=3, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample \n \n $\mathbf{(b)}$')  

print(ode_pers)   	
plt.show()     			

#plot dterministic simulation    	
simulationPoint = [49.61, 1.43, 4.4, 21.83, 1.72, 0.78, 123.12, 2.2, 120.12] 
model1.getPerAmp(simulationPoint, mode="ode", indx=1)   

region1 = solver.Region(viablePoints1, model1, "region1")     
region2 = solver.Region(viablePoints2, model2, "region2")     

region1PCA = pca.transform(viablePoints1)
region2PCA = pca.transform(viablePoints2)  

#plot overlap of both regions 
plt.scatter(region2PCA[:, 0], region2PCA[:, 1], c="#15A357", alpha=1, edgecolor="#158257",  rasterized=True)  
plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E74C8", alpha=0.5, edgecolor="#0E56C8",  rasterized=True)
plt.xlabel('PC 1')
plt.ylabel('PC 2') 
plt.show()
#plot points on pca projections  
filesA = ["represViableSet_IterGA.p", "repres_Region0ViableSet_Iter1.p", "repres_Region0ViableSet_Iter9.p"] 
filesB = ["repres_Region0CandidateSet_Iter1.p", "repres_Region0CandidateSet_Iter2.p", "repres_Region0CandidateSet_Iter10.p"] 

pca1 = PCA(n_components=2) 
pca2 = PCA(n_components=2) 

pointsa = pickle.load(open(path.join(folder1, filesA[0]) , "rb"))
pointsb = pickle.load(open(path.join(folder2, filesA[0]) , "rb")) 

pca1.fit(pointsa) 
pca2.fit(pointsb)  

#plot viable regions in PCA space     
plt.rcParams.update({'font.size': 12})
i = 1
iters = [1, 2, 10] 
for filea, fileb in zip(filesA, filesB):	

	file1a =  path.join(folder1, filea)     
	file2a =  path.join(folder2, filea)    
	file1b =  path.join(folder1, fileb)     
	file2b =  path.join(folder2, fileb)   	

	viablePoints1 = pickle.load(open(file1a, "rb"))    
	viablePoints2 = pickle.load(open(file2a, "rb"))  
	
	candidatePoints1 = pickle.load(open(file1b, "rb"))  
	candidatePoints2 = pickle.load(open(file2b, "rb"))

	region1PCA = pca1.transform(viablePoints1) 
	region2PCA = pca2.transform(viablePoints2) 
	
	region1PCAcandidates = pca1.transform(candidatePoints1) 
	region2PCAcandidates = pca2.transform(candidatePoints2)   

	ax1=plt.subplot(2, 3, i) 
	ax1.title.set_text("Iteration " + str(iters[i-1]))

	plt.scatter(region1PCAcandidates[:, 0], region1PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)  
	plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)    
	if i == 1:
		plt.ylabel('PC 2')  
	if i == 2:
		plt.xlabel(r"$\bf{(a)}}$")   	 	

	ax1=plt.subplot(2, 3, i + 3) 
	plt.scatter(region2PCAcandidates[:, 0], region2PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)   	
	plt.scatter(region2PCA[:, 0], region2PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)  
	if i == 2:
		plt.xlabel("PC 1 \n"+ r"$\bf{(c)}$") 
	else:
		plt.xlabel("PC 1")   
	if i == 1:
		plt.ylabel('PC 2')    
	
	i += 1
plt.show()	 
 

solver_object = solver.Solver(model1)
viableSets = [region1, region2]   		
paramNames = [r"$\alpha$", r"$\alpha_0$",r"$n$", r"$\beta$",r"$\delta_m$", r"$\delta_p$",r"$Kd$",r"$m$"] 
units = [r"[$h^{-1}$]", r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}]$",r"[$nM$]",""]    
solver_object.plotParameterVariances(viableSets, names=paramNames, units=units)         


#plot few repressilator responses 	  
print("plotting model 2") 	
number = np.size(viablePoints1, 0)   
rndPoints = np.random.randint(number, size=10)  
for pnt in rndPoints:     
	print(viablePoints1[pnt,:])     
	model1.plotModel(viablePoints1[pnt,:]) 
	model1.plotModel(viablePoints1[pnt,:], mode="ssa")      	
print("plotting model 2")  	
number = np.size(viablePoints2, 0)     
rndPoints = np.random.randint(number, size=10)   
for pnt in rndPoints:    
	model2.plotModel(viablePoints2[pnt,:])   
	model2.plotModel(viablePoints2[pnt,:], mode="ssa")    	


#calculate viable volume for both cases and determine deviations 
solver_object = solver.Solver(model1) 
vol1 = solver_object.getViableVolume([region1], 100000) 
solver_object = solver.Solver(model2)    
vol2 = solver_object.getViableVolume([region2], 100000)      

print("Region 2 is " + str(vol2/vol1) + "greater than Region 1.")      
