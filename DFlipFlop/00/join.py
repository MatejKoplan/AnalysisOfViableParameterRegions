import pickle  
import numpy as np 

"""
Joins granulated regions to single set for the purposes of analysis and visualization. 
"""

filesA = ["flipflopViableSet_IterGA.p", "flipflop_Region0ViableSet_Iter1.p", "flipflop_Region0ViableSet_Iter9.p"]  
filesB = ["flipflop_Region0CandidateSet_Iter1.P", "flipflop_Region0CandidateSet_Iter2.P", "flipflop_Region0CandidateSet_Iter10.P"]  

#iteration 1
candidate00Iter1 = pickle.load(open("flipflop_Region00CandidateSet_Iter1.p", "rb")) 
candidate01Iter1 = pickle.load(open("flipflop_Region01CandidateSet_Iter1.p", "rb"))  

candidateIter1 = np.vstack((candidate00Iter1, candidate01Iter1)) 
pickle.dump(candidateIter1, open("flipflop_Region0CandidateSet_Iter1.p", "wb+"))  
      
candidate00Iter2 = pickle.load(open("flipflop_Region00CandidateSet_Iter2.p", "rb")) 
candidate01Iter2 = pickle.load(open("flipflop_Region01CandidateSet_Iter2.p", "rb"))  

candidateIter2 = np.vstack((candidate00Iter2, candidate01Iter2)) 
pickle.dump(candidateIter2, open("flipflop_Region0CandidateSet_Iter2.p", "wb+"))   

candidate00Iter10 = pickle.load(open("flipflop_Region00CandidateSet_Iter10.p", "rb"))  
candidate01Iter10 = pickle.load(open("flipflop_Region01CandidateSet_Iter10.p", "rb"))   

candidateIter10 = np.vstack((candidate00Iter10, candidate01Iter10)) 
pickle.dump(candidateIter10, open("flipflop_Region0CandidateSet_Iter10.p", "wb+"))    


viable00Iter1 = pickle.load(open("flipflop_Region00ViableSet_Iter1.p", "rb")) 
viable01Iter1 = pickle.load(open("flipflop_Region01ViableSet_Iter1.p", "rb")) 

viable0Iter1 = np.vstack((viable00Iter1, viable01Iter1)) 
pickle.dump(viable0Iter1, open("flipflop_Region0ViableSet_Iter1.p", "wb+")) 


viable00Iter9 = pickle.load(open("flipflop_Region00ViableSet_Iter9.p", "rb")) 
viable01Iter9 = pickle.load(open("flipflop_Region01ViableSet_Iter9.p", "rb")) 

viable0Iter9 = np.vstack((viable00Iter9, viable01Iter9))   
pickle.dump(viable0Iter9, open("flipflop_Region0ViableSet_Iter9.p", "wb+"))  


viable00Iter10 = pickle.load(open("flipflop_Region00ViableSet_Iter10.p", "rb")) 
viable01Iter10 = pickle.load(open("flipflop_Region01ViableSet_Iter10.p", "rb")) 

viable0Iter10 = np.vstack((viable00Iter10, viable01Iter10))    
pickle.dump(viable0Iter10, open("flipflop_Region0ViableSet_Iter10.p", "wb+")) 