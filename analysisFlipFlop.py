import solver
import pickle
import os.path as path
from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
from dFlipFlop import DFlipFlop    
from deap import creator, base, tools, algorithms  

parameter_values = {  "transcription": {"min": 0.01, "max": 50},  
			"translation": {"min": 0.01, "max": 50}, 
			"protein_production": {"min": 0.1, "max": 50},            				
			"rna_degradation": {"min": 0.1, "max": 100},        
			"protein_degradation": {"min": 0.001, "max": 50},         
			"hill": {"min": 1, "max": 5},         
			"Kd": {"min": 0.01, "max": 250}, 
			"protease_concentration": {"min": 10, "max":1000}      	
			}          


creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Candidate", list, fitness=creator.FitnessMax)	 	
toolbox = base.Toolbox()	  
toolbox.register("candidate", solver.Solver.generateCandidate)  
	
plt.rcParams.update({'font.size': 12})  
	
def getModel(omega1, omega2):  					
	params = ["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation", "protein_degradation", "Kd","hill"]
	
	if omega2 == 0: 
		params.append("Kd") 
		params.append("protease_concentration")  
		
	return DFlipFlop(parameter_values, np.array(params), np.array([0.0 , 0.0, 100.0, 0.0]), omega1=omega1, omega2=omega2)       

#get models 	
folder00 = path.join(".", "DFlipFlop", "00")  
folder01 = path.join(".", "DFlipFlop", "01") 
folder10 = path.join(".", "DFlipFlop", "10") 
folder11 = path.join(".", "DFlipFlop", "11")
folders = [folder00, folder01, folder10, folder11]

model00 = getModel(0, 0) 
model01 = getModel(0, 1)
model10 = getModel(1, 0)
model11 = getModel(1, 1)
models = [model00, model01, model10, model11]  
	 					
file00 =  path.join(folder00, "flipflop_Region0ViableSet_Iter10.p")               
file01 =  path.join(folder01, "flipflop_Region0ViableSet_Iter10.p")  
file10 =  path.join(folder10, "flipflop_Region0ViableSet_Iter10.p")  
file11 =  path.join(folder11, "flipflop_Region0ViableSet_Iter10.p")  

viablePoints00 = pickle.load(open(file00, "rb"))
viablePoints01 = pickle.load(open(file01, "rb"))
viablePoints10 = pickle.load(open(file10, "rb"))
viablePoints11 = pickle.load(open(file11, "rb"))    

region00 = solver.Region(viablePoints00, model00, "flipflop region")
region01 = solver.Region(viablePoints01, model01, "flipflop region")
region10 = solver.Region(viablePoints10, model10, "flipflop region")
region11 = solver.Region(viablePoints11, model11, "flipflop region")


pca00 = PCA(n_components=2)
pca01 = PCA(n_components=2)
pca10 = PCA(n_components=2)
pca11 = PCA(n_components=2)

pca00.fit(viablePoints00)
pca01.fit(viablePoints01) 
pca10.fit(viablePoints10) 
pca11.fit(viablePoints11)   
 
solver_object00 = solver.Solver(model00)
solver_object01 = solver.Solver(model01)
solver_object10 = solver.Solver(model10)
solver_object11 = solver.Solver(model11) 

#calculate viable volumes
print("Viable volume for 00")
solver_object00.getViableVolume([region00], 100000) 
print("Viable volume for 01")   
solver_object01.getViableVolume([region01], 100000)  
print("Viable volume for 10")    
solver_object10.getViableVolume([region10], 100000)  
print("Viable volume for 11")   
solver_object11.getViableVolume([region11], 100000)        

filesA = ["flipflopViableSet_IterGA.p", "flipflop_Region0ViableSet_Iter1.p", "flipflop_Region0ViableSet_Iter9.p"]  
filesB = ["flipflop_Region0CandidateSet_Iter1.P", "flipflop_Region0CandidateSet_Iter2.P", "flipflop_Region0CandidateSet_Iter10.P"]   


#plot viable regions in PCA space  
i = 1
for model, folder in zip(models, folders):  
 
	pca = PCA(n_components=2)  
	points = pickle.load(open(path.join(folder, filesA[0]) , "rb")) 
	pca.fit(points)   

	iters = [1, 2, 10]
	for filea, fileb in zip(filesA, filesB):	 

		file1a =  path.join(folder, filea)     
		file1b =  path.join(folder, fileb)    
		
		viablePoints1 = pickle.load(open(file1a, "rb"))     
		candidatePoints1 = pickle.load(open(file1b, "rb"))    

		region1PCA = pca.transform(viablePoints1) 	
		region1PCAcandidates = pca.transform(candidatePoints1) 

		ax1=plt.subplot(4, 3, i)
		if i < 4:
			ax1.title.set_text("Iteration " + str(iters[i%3-1]))

		plt.scatter(region1PCAcandidates[:, 0], region1PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)   
		plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)    
		
		if i == 1:
			plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 0}$ \n  PC 2')
		if i == 4:
			plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 1}$ \n  PC 2')
		if i == 7:
			plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 0}$ \n  PC 2')
		if i == 10: 
			plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 1}$ \n  PC 2') 
		if i == 10 or i == 11 or i == 12:
			plt.xlabel('PC 1')
			
		i += 1
		
plt.show()	
 
 
params = ["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation", "protein_degradation", "Kd","hill", "Kd", "protease_concentration"] 
paramNames = [r"$\alpha_1$", r"$\alpha_2$",r"$\alpha_3$", r"$\alpha_4$",r"$\delta_1$", r"$\delta_2$",r"$Kd$", r"$n$", r"$K_M$", r"$E$"]     
units = [r"[$h^{-1}$]", r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}]$",r"[$nM$]", "", r"[$nM$]", r"[$nM$]"] 
 

colors = ["#15A357", "#0E74C8", "#A62B21", "black"]  
colorLen = len(colors)
figure = plt.figure() 

#plot parameter ranges 
i = 0
for param_name, unit in zip(paramNames, units):  
	plt.subplot(3,4, i + 1) 	
		
	currRegions = 4
	if i > 7:
		values = [viablePoints00[:,i], [0], viablePoints10[:,i], [0]] 
		currRegions = 2
	else: 
		values = [viablePoints00[:,i], viablePoints01[:,i], viablePoints10[:,i], viablePoints11[:,i]]
		
	bp = plt.boxplot(values)     
	plt.ylabel(param_name + " " + unit)    	
	
	for k in range(4):  
		col = colors[k % colorLen ]  	 	 
		plt.setp(bp['boxes'][k], color=col, linewidth=1.5)    
		plt.setp(bp['caps'][2*k], color=col, linewidth=1.5)  
		plt.setp(bp['caps'][2*k + 1], color=col, linewidth=1.5) 
		plt.setp(bp['whiskers'][2*k], color=col, linewidth=1.5)  
		plt.setp(bp['whiskers'][2*k + 1], color=col, linewidth=1.5)   
		plt.setp(bp['fliers'][k], color=col) 
		plt.setp(bp['medians'][k], color=col, linewidth=1.5)  

	i = i + 1
	
allBoxes = bp['boxes'] 
allNames = ["$\Omega_1 = 0, \Omega_2 = 0$", "$\Omega_1 = 0, \Omega_2 = 1$", "$\Omega_1 = 1, \Omega_2 = 0$", "$\Omega_1 = 1, \Omega_2 = 1$"] 			
#draw legend 
figure.legend(allBoxes, allNames, 'lower right')
plt.show()   


###                   ###
###  SSA simulations  ###
###                   ### 

readFromFile = False                        
numSamples = 3

if readFromFile:     
	sampleNumbers00 = pickle.load(open("ssa_samples_flipflop00", "rb"))
	sampleNumbers01 = pickle.load(open("ssa_samples_flipflop01", "rb"))  
	sampleNumbers10 = pickle.load(open("ssa_samples_flipflop10", "rb"))  	
	sampleNumbers11 = pickle.load(open("ssa_samples_flipflop11", "rb"))   
else:
	num00 = viablePoints00.shape[0] 
	num01 = viablePoints01.shape[0] 
	num10 = viablePoints10.shape[0] 
	num11 = viablePoints11.shape[0] 
		
	sampleNumbers00 = np.random.choice(num00, numSamples, replace=False)
	sampleNumbers01 = np.random.choice(num01, numSamples, replace=False)
	sampleNumbers10 = np.random.choice(num10, numSamples, replace=False)
	sampleNumbers11 = np.random.choice(num11, numSamples, replace=False)

	pickle.dump(sampleNumbers00, open("ssa_samples_flipflop00", "wb+"))   	
	pickle.dump(sampleNumbers01, open("ssa_samples_flipflop01", "wb+"))   
	pickle.dump(sampleNumbers10, open("ssa_samples_flipflop10", "wb+"))   
	pickle.dump(sampleNumbers11, open("ssa_samples_flipflop11", "wb+"))   

if readFromFile:
	samples00 = pickle.load(open("ssa_samples_flipflop00","rb"))
	samples01 = pickle.load(open("ssa_samples_flipflop01","rb"))
	samples10 = pickle.load(open("ssa_samples_flipflop10","rb")) 
	samples11 = pickle.load(open("ssa_samples_flipflop11","rb")) 
else: 	
	samples00 = viablePoints00[sampleNumbers00,:]
	samples01 = viablePoints01[sampleNumbers01,:]
	samples10 = viablePoints10[sampleNumbers10,:]
	samples11 = viablePoints11[sampleNumbers11,:]	

	pickle.dump(samples00, open("ssa_samples_flipflop00", "wb+"))    
	pickle.dump(samples01, open("ssa_samples_flipflop01", "wb+")) 
	pickle.dump(samples10, open("ssa_samples_flipflop10", "wb+")) 
	pickle.dump(samples11, open("ssa_samples_flipflop11", "wb+")) 
	
#plot few simulations ODE + SSA  
t = np.linspace(0, model00.T, model00.N) 

ts_total_stochastic00 = [] 
Y_total_stochastic00 = []
t_total_ode00 = []
Y_total_ode00 = []

ts_total_stochastic01 = [] 
Y_total_stochastic01 = []
t_total_ode01 = []
Y_total_ode01 = []

ts_total_stochastic10 = [] 
Y_total_stochastic10 = []
t_total_ode10 = []
Y_total_ode10 = []  

ts_total_stochastic11 = [] 
Y_total_stochastic11 = []
t_total_ode11 = []
Y_total_ode11 = []

if readFromFile:
	t_total_ode00, Y_total_ode00, ts_total_stochastic00, Y_total_stochastic00 = pickle.load(open("ssa_simulations_flipflop00", "rb")) 
	t_total_ode01, Y_total_ode01, ts_total_stochastic01, Y_total_stochastic01 = pickle.load(open("ssa_simulations_flipflop01", "rb")) 
	t_total_ode10, Y_total_ode10, ts_total_stochastic10, Y_total_stochastic10 = pickle.load(open("ssa_simulations_flipflop10", "rb")) 
	t_total_ode11, Y_total_ode11, ts_total_stochastic11, Y_total_stochastic11 = pickle.load(open("ssa_simulations_flipflop11", "rb"))  

for i in range(numSamples):   
	sample00 = samples00[i]
	sample01 = samples01[i]
	sample10 = samples10[i]
	sample11 = samples11[i]

	if readFromFile:
		ts00 = ts_total_stochastic00[i]   
		Y00 = Y_total_stochastic00[i]  
		Y_ode00 = Y_total_ode00[i]  
		t00 = t_total_ode00[i]

		ts01 = ts_total_stochastic01[i]   
		Y01 = Y_total_stochastic01[i]  
		Y_ode01 = Y_total_ode01[i]  
		t01 = t_total_ode01[i]

		ts10 = ts_total_stochastic10[i]   
		Y10 = Y_total_stochastic10[i]  
		Y_ode10 = Y_total_ode10[i]  
		t10 = t_total_ode10[i] 

		ts11 = ts_total_stochastic11[i]    
		Y11 = Y_total_stochastic11[i]  
		Y_ode11 = Y_total_ode11[i]  
		t11 = t_total_ode11[i]		
	else: 
	
		ts00, Y00 = model00.flipflopStochastic(sample00)  
		Y00 = Y00/model00.omega 
		Y_ode00 = model00.simulate(sample00)  
		ts_total_stochastic00.append(ts00)
		Y_total_stochastic00.append(Y00) 
		t_total_ode00.append(t)  
		Y_total_ode00.append(Y_ode00)  
		
		ts01, Y01 = model01.flipflopStochastic(sample01)  
		Y01 = Y01/model01.omega 
		Y_ode01 = model01.simulate(sample01)  
		ts_total_stochastic01.append(ts01)
		Y_total_stochastic01.append(Y01) 
		t_total_ode01.append(t)  
		Y_total_ode01.append(Y_ode01)

		ts10, Y10 = model10.flipflopStochastic(sample10)  
		Y10 = Y10/model10.omega 
		Y_ode10 = model10.simulate(sample10)  
		ts_total_stochastic10.append(ts10)
		Y_total_stochastic10.append(Y10) 
		t_total_ode10.append(t)  
		Y_total_ode10.append(Y_ode10)

		ts11, Y11 = model11.flipflopStochastic(sample11)  
		Y11 = Y11/model11.omega 
		Y_ode11 = model11.simulate(sample11)  
		ts_total_stochastic11.append(ts11)
		Y_total_stochastic11.append(Y11) 
		t_total_ode11.append(t)  
		Y_total_ode11.append(Y_ode11)   		

		
	ax1=plt.subplot(4, 3, i + 1)
	ax1.set_title("Sample " + str(i + 1), fontsize=12)
	
	lines = plt.plot(t, Y_ode00[:,2], ts00, Y00[:,2], t, Y_ode00[:,3], ts00, Y00[:,3])   
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#A62B21")
	plt.setp(lines[1], linewidth=1.5, alpha = 0.65, c="#A62B21") 
	plt.setp(lines[2], linestyle = "--", linewidth=1.5, alpha=1, c="#0E74C8")
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#0E74C8")  

	if i == 0: 
		plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 0}$ \n  Concentration [$nM$]')   	
	
	ax1=plt.subplot(4, 3, i + 4)  
	ax1.set_title("Sample " + str(i + 1), fontsize=12)    
	lines = plt.plot(t, Y_ode01[:,2], ts01, Y01[:,2], t, Y_ode01[:,3], ts01, Y01[:,3])   
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#A62B21")
	plt.setp(lines[1], linewidth=1.5, alpha = 0.65, c="#A62B21") 
	plt.setp(lines[2], linestyle = "--", linewidth=1.5, alpha=1, c="#0E74C8")
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#0E74C8") 

	if i == 0: 
		plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 1}$ \n Concentration [$nM$]') 
	
	ax1=plt.subplot(4, 3, i + 7) 
	ax1.set_title("Sample " + str(i + 1), fontsize=12) 
	lines = plt.plot(t, Y_ode10[:,2], ts10, Y10[:,2], t, Y_ode10[:,3], ts10, Y10[:,3])   
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#A62B21")
	plt.setp(lines[1], linewidth=1.5, alpha = 0.65, c="#A62B21") 
	plt.setp(lines[2], linestyle = "--", linewidth=1.5, alpha=1, c="#0E74C8")
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#0E74C8") 
	
	if i == 0: 
		plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 0}$ \n  Concentration [$nM$]') 
	
	ax1=plt.subplot(4, 3, i + 10)
	ax1.set_title("Sample " + str(i + 1), fontsize=12) 	
	lines = plt.plot(t, Y_ode11[:,2], ts11, Y11[:,2], t, Y_ode11[:,3], ts11, Y11[:,3])     
	plt.setp(lines[0], linestyle = "--", linewidth=1.5, alpha=1, c="#A62B21")
	plt.setp(lines[1], linewidth=1.5, alpha = 0.65, c="#A62B21") 
	plt.setp(lines[2], linestyle = "--", linewidth=1.5, alpha=1, c="#0E74C8")
	plt.setp(lines[3], linewidth=1.5, alpha = 0.65, c="#0E74C8")  

	plt.xlabel("Time [$h$]")     
		
	if i == 0: 
		plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 1}$ \n  Concentration [$nM$]') 

plt.show()    

if not readFromFile:
	pickle.dump((t_total_ode00, Y_total_ode00, ts_total_stochastic00, Y_total_stochastic00), open("ssa_simulations_flipflop00", "wb+"))  
	pickle.dump((t_total_ode01, Y_total_ode01, ts_total_stochastic01, Y_total_stochastic01), open("ssa_simulations_flipflop01", "wb+")) 
	pickle.dump((t_total_ode10, Y_total_ode10, ts_total_stochastic10, Y_total_stochastic10), open("ssa_simulations_flipflop10", "wb+")) 
	pickle.dump((t_total_ode11, Y_total_ode11, ts_total_stochastic11, Y_total_stochastic11), open("ssa_simulations_flipflop11", "wb+"))	

repetitions = 100         
amps00 = []
pers00 = []  
ode_amps00 = []
ode_pers00 = []  

amps01 = []
pers01 = []  
ode_amps01 = []
ode_pers01 = []		
		
amps10 = []
pers10 = []  
ode_amps10 = []
ode_pers10 = []

amps11 = []
pers11 = []  
ode_amps11 = []
ode_pers11 = [] 

if readFromFile:      
	pers00, amps00 = pickle.load(open("ssa_persamps_flipflop00","rb"))  
	pers01, amps01 = pickle.load(open("ssa_persamps_flipflop01","rb"))  
	pers10, amps10 = pickle.load(open("ssa_persamps_flipflop10","rb"))  
	pers11, amps11 = pickle.load(open("ssa_persamps_flipflop11","rb"))

for i in range(numSamples):
	sample00 = samples00[i]
	per00, amp00 = model00.getPerAmp(sample00, mode="ode", indx1=2, indx2=3)   
	ode_amps00.append(amp00)   
	ode_pers00.append(per00)  	

	sample01 = samples01[i]
	per01, amp01 = model01.getPerAmp(sample01, mode="ode", indx1=2, indx2=3)   
	ode_amps01.append(amp01)   
	ode_pers01.append(per01) 
	
	sample10 = samples10[i]
	per10, amp10 = model10.getPerAmp(sample10, mode="ode", indx1=2, indx2=3)   
	ode_amps10.append(amp10)   
	ode_pers10.append(per10) 

	sample11 = samples11[i]
	per11, amp11 = model11.getPerAmp(sample11, mode="ode", indx1=2, indx2=3)   
	ode_amps11.append(amp11)   
	ode_pers11.append(per11) 

	if not readFromFile:      	
		curra00 = []
		currp00 = []
		for j in range(repetitions):
			per00, amp00 = model00.getPerAmp(sample00, mode="ssa", indx1=2, indx2=3)  
			amp00 = amp00/model00.omega    
			curra00.append(amp00)  
			currp00.append(per00)   		
		amps00.append(curra00) 
		pers00.append(currp00)   

		curra01 = []
		currp01 = []
		for j in range(repetitions):
			per01, amp01 = model01.getPerAmp(sample01, mode="ssa", indx1=2, indx2=3)  
			amp01 = amp01/model01.omega    
			curra01.append(amp01)  
			currp01.append(per01)   		
		amps01.append(curra01) 
		pers01.append(currp01) 
		
		curra10 = []
		currp10 = []
		for j in range(repetitions):
			per10, amp10 = model10.getPerAmp(sample10, mode="ssa", indx1=2, indx2=3)  
			amp10 = amp10/model10.omega    
			curra10.append(amp10)  
			currp10.append(per10)   		
		amps10.append(curra10) 
		pers10.append(currp10) 

		curra11 = []
		currp11 = []
		for j in range(repetitions):
			per11, amp11 = model11.getPerAmp(sample11, mode="ssa", indx1=2, indx2=3)    
			amp11 = amp11/model11.omega    
			curra11.append(amp11)  
			currp11.append(per11)      		
		amps11.append(curra11) 
		pers11.append(currp11) 			

if not readFromFile:       	
	pickle.dump((pers00, amps00), open("ssa_persamps_flipflop00", "wb+"))  
	pickle.dump((pers01, amps01), open("ssa_persamps_flipflop01", "wb+"))    
	pickle.dump((pers10, amps10), open("ssa_persamps_flipflop10", "wb+"))   
	pickle.dump((pers11, amps11), open("ssa_persamps_flipflop11", "wb+"))    	



plt.subplot(4,2,1)	

bp = plt.boxplot(amps00)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)      
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_amps00[i], ode_amps00[i]], linewidth=3, color="black")
	
plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 0}$ \n Amplitude [$nM$]')  
plt.xlabel('Sample')
plt.subplot(4,2,3)	

bp = plt.boxplot(amps01)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_amps01[i], ode_amps01[i]], linewidth=3, color="black")
	
plt.ylabel('$\mathbf{\Omega_1 = 0, \Omega_2 = 1}$ \n Amplitude [$nM$]') 
plt.xlabel('Sample')
plt.subplot(4,2,5)	

bp = plt.boxplot(amps10)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_amps10[i], ode_amps10[i]], linewidth=3, color="black")
	
plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 0}$ \n Amplitude [$nM$]') 
plt.xlabel('Sample')
plt.subplot(4,2,7) 	

bp = plt.boxplot(amps11)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_amps11[i], ode_amps11[i]], linewidth=3, color="black") 
	
plt.ylabel('$\mathbf{\Omega_1 = 1, \Omega_2 = 1}$ \n Amplitude [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(a)}$') 



plt.subplot(4,2,2)  	
bp = plt.boxplot(pers00)    
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_pers00[i], ode_pers00[i]], linewidth=3, color="black") 
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample')  

plt.subplot(4,2,4)  	 
bp = plt.boxplot(pers01)    
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_pers01[i], ode_pers01[i]], linewidth=3, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample')  

plt.subplot(4,2,6)  	
bp = plt.boxplot(pers10)    
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_pers10[i], ode_pers10[i]], linewidth=3, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample')  


plt.subplot(4,2,8)  	
bp = plt.boxplot(pers11)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.15, i + 1 + 0.15], [ode_pers11[i], ode_pers11[i]], linewidth=3, color="black") 
	
plt.ylabel('Period [$h$]')  
plt.xlabel('Sample \n \n $\mathbf{(b)}$')   
	
plt.show()   