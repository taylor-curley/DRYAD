# -*- coding: utf-8 -*-
"""
DRYAD simulation.

Created by Aaron Benjamin, University of Illinois at Urbana-Champaign.
Reference: Benjamin, A. S. (2011). Representational explanations
of process dissociations in recognition: The DRYAD theory of 
aging and memory judgments.

Modified for Python by Taylor Curley (Georgia Tech)
    taylor.curley@gatech.edu
    
Last updated: September, 2018
"""

import numpy as np
from dryad_modules import create_stimuli,learn,create_test,compute_parameters
from dryad_modules import Icompute,recall_contexts,choose_contexts,response

np.seterr(divide='ignore',invalid='ignore')

"""
Initialize simulation parameters
"""

# Maximum and minimum correlation between the two context vectors
# The program can grind to a halt if these are set too conservatively.
maxR = 0.5
minR = -0.5

# Starting, ending, and step-size values for decision criterion
CriterionBegin = 0.05
CriterionEnd = 0.05
CriterionStep = 0.05
if CriterionBegin == CriterionEnd:
    Criterion_list = [CriterionBegin]
else:
    Criterion_list = np.arange(CriterionBegin,CriterionEnd+.01,CriterionStep)

# Total number of study items (must be divisible by 2)
NumStudyItems = 40

# Number of different contexts (works for 2)
NumContexts = 2

# Number of (new) distractors on the exclusion test
NumDistract = 40

# Number of subjects for the simulation for each set of parameter values.
NumSubjects = 10

# Starting, ending, and step-size value for memory fidelity factor (L).
# This parameter is called "F" in the 2011 paper.
Lbegin = 0.1
Lend = 0.9
Lstep = 0.1
if Lbegin == Lend:
    Llist = [Lbegin]
else:  
    Llist = np.arange(Lbegin,Lend+.1,Lstep)
    
# Starting, ending, and step-size value for the number of dimensions
# accorded to the representation of the "ITEM" (I)
InodeBegin = 20
InodeEnd = 20
InodeStep = 5
if InodeBegin == InodeEnd:
    Inode_list = [InodeBegin]
else:
    Inode_list = np.arange(InodeBegin,InodeEnd+1,InodeStep)

# Starting, ending, and step-size value for the number of dimensions
# accorded to the representation of the "CONTEXT" (C)
CnodeBegin = 2
CnodeEnd = 2
CnodeStep = 2
if CnodeBegin == CnodeEnd:
    Cnode_list = [CnodeBegin]
else:
    Cnode_list = np.arange(CnodeBegin,CnodeEnd+1,CnodeStep)

"""
Progress bar 

Set "see_progress_bar" to "0" to suppress progress bar.
Requires installation of "progressbar2" in pip/conda. 
"""
see_progress_bar = 1

if see_progress_bar == 1:
    import progressbar
    bar = progressbar.ProgressBar(max_value=NumSubjects*len(Criterion_list)*
                                  len(Inode_list)*len(Cnode_list)*len(Llist))

"""
Set up dataframe
"""

import pandas as pd

columns = ["Cparam","Iparam","LParam","Iteration","HR","FARtbx","FARnew","FARnew_F"]

fileData = pd.DataFrame(columns = columns)

iter = 0

"""
Begin simulation
"""

for Criterion in Criterion_list:
    for Inodes in Inode_list:
        for Cnodes in Cnode_list:
            for L in Llist:
                for i in range(NumSubjects):
                    # Create to-be-learned stimuli
                    stimuli,contexts,indexContextOne,indexContextTwo = create_stimuli(Inodes,Cnodes,NumStudyItems,NumContexts,minR,maxR)
                    
                    # Encode stimuli into the model
                    memory = learn(stimuli[:,1:], L)
                    
                    # Create test list (add new distractors to studied items and remove contexts)
                    test = create_test(stimuli,NumDistract,Inodes,Cnodes)
                    
                    # Compute familiarity (Mg in paper) and compare to criterion
                    A,similarity = Icompute(memory[:,0:],test[:,1:])
                    
                    # Evaluate the familiarity - if not above criterion, designate as "NaN"
                    familiarity = similarity[:]
                    familiarity[familiarity>Criterion]=1
                    familiarity[familiarity<Criterion]="nan"
                    
                    # Recall items ("e" in paper) and choose most likely context
                    C = recall_contexts(memory[:,0:],test[:,1:],A,NumContexts)
                    
                    DP,S = choose_contexts(C, contexts, NumContexts, Cnodes)
                    
                    # Combine familiarity and recollected context into a decision
                    R = response(S,familiarity)
                    
                    # General HR, FA-TBX, and FAR-New for this subject and place values into
                    # matrix D.
                    HR,FARtbx,FARnew,FARnew_F = compute_parameters(familiarity, R, NumStudyItems, NumDistract, indexContextOne, indexContextTwo)
                    
                    # Progress bar
                    if see_progress_bar == 1:
                        bar.update(iter+1)
                    
                    # Write to dataframe
                    toFile = [Cnodes,Inodes,L,i,HR,FARtbx,FARnew,FARnew_F]
                    fileData.loc[iter] = toFile
                    
                    iter = iter+1
                    
                    
"""
Write results to file
"""
fileData.to_csv("/path/to/file.csv")
