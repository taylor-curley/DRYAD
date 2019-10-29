# -*- coding: utf-8 -*-
"""
DRYAD modules.

Created by Aaron Benjamin, University of Illinois at Urbana-Champaign.
Reference: Benjamin, A. S. (2011). Representational explanations
of process dissociations in recognition: The DRYAD theory of 
aging and memory judgments.

Modified for Python by Taylor Curley (Georgia Tech)
    taylor.curley@gatech.edu
    
Last updated: September, 2018
"""

import numpy as np

"""
Functions
"""
def create_stimuli(Inodes, Cnodes, NumItems, NumContexts, minR, maxR):
    
    """
    create_stimuli performs the following functions: 
    1) Generates random vectors of numbers (of -1, 0, 1) to represent stimuli.
    2) Creates context vectors that are correlated to a given degree.
    3) Randomly distributes the contexts to the items.
    4) Writes the indexes of items that received a certain context.
    
    Parameters
    ----------
    Inodes : integer
        Integer specifying the length of an "item" vector.
    Cnodes : integer
        Integer specifying the length of a "context" vector.
    NumItems : integer
        Integer specifying the number of items at study (default - 40).
    NumContexts : integer
        Integer specifying the number of contexts (default - 2).
    minR : float
        Float specifying the minumum correlation between two context vectors.
    maxR : float
        Float specifying the maximum correlation between two context vectors.

    Returns
    ----------
    stimuli : matrix
        Matrix of numbered study items of dimensions NumItems x Inodes+Cnodes.
    contexts : matrix
        Matrix of context vectors of dimensions NumContexts x Cnodes.
    indexContextOne : vector
        Vector specifying the location of Context 1 items in the study list.
    indexContextTwo : vector
        Vector specifying the location of Context 2 items in the study list.
    """    
    
    Tnodes = Inodes + Cnodes
    
    # Evaluate correctness of numbers of contexts and study items
    
    # Create random matrix of values -1 to 1 with dimensions
    # Number of Items x Item Dimensions
    stimuli = np.random.randint(-1,2,(NumItems, Inodes))
    
    # Create contexts (formerly "subCreate_Contexts" in original code)
    cmat = np.matrix(((-1, -1), (-1, -1)))
    while cmat[0,1] < minR or cmat[0,1] > maxR:
        contexts = np.random.randint(-1,2,(NumContexts, Cnodes))
        cmat = np.corrcoef(contexts)

    # Distribute contexts to items
    # Random distribution right now - will work on even distribution of contexts
    # in later code.
    stimuli2 = np.zeros((NumItems,Tnodes))
    indexContextOne = []
    indexContextTwo = []
    for i in range(NumItems):
        j = int(np.random.choice(NumContexts,1))
        stimuli2[i] = np.hstack((stimuli[i,:],contexts[j,:]))
        if j == 0:
            indexContextOne.append(i)
        else:
            indexContextTwo.append(i)
    
    # Number the study trials
    stimuli = np.hstack((np.arange(1,len(stimuli2)+1,1).reshape(len(stimuli2),1),stimuli2))
    return stimuli,contexts,indexContextOne,indexContextTwo
    

def learn(stimuli, L):
    
    """
    The learn function writes an event as a memory trace by degrading the event 
    vector by randomly changing each features to a zero with probability (1-L).
    
    Parameters
    ----------
    stimuli : matrix
        A matrix consisting of all original studied memory item vectors.
    L : float
        The forgetting rate for the studied items.

    Returns
    ----------
    memory : matrix
        A matrix of the degraded memory trace vectors, as given by the forgetting
        parameter.
    """

    memory = np.zeros(np.shape(stimuli))
    for i in range(np.shape(stimuli)[0]):
        for j in range(np.shape(stimuli)[1]):
            if np.random.random_sample(1) > L:
                #memory[i,j] = np.random.randint(-1,2)
                memory[i,j] = 0
            else:
                memory[i,j] = stimuli[i,j]
    return memory

def create_test(stimuli,NumDistract,Inodes,Cnodes):
    
    """
    Creates a numbered list of items for test consisting of "old" item vectors 
    ("stimuli") and "new" item vectors.
    
    Parameters
    ----------
    stimuli : matrix
        A matrix consisting of all original studied memory item vectors.
    NumDistract : integer
        The number of new distractor items for the test.
    Inodes : integer
        Specifies the length of the vectors for the new distractor items.
    Cnodes : integer
        Specifies the length of the blank context mini vector.

    Returns
    ----------
    A numbered matrix of old (studied) and new (unstudied) memory item vectors.
    """
    
    studied = stimuli[:,1:Inodes+1]
    #unstudied = np.hstack((np.random.randint(-1,2,(NumDistract,Inodes)),np.zeros((NumDistract,Cnodes))))
    unstudied = np.random.randint(-1,2,(NumDistract, Inodes))
    test = np.vstack((studied,unstudied))
    
    # Number the test items
    return np.hstack((np.arange(1,len(test)+1,1).reshape(len(test),1),test))


def vectorSimilarity(inputVector1,inputVector2):
    
    """
    vectorSimilarity caculates the similarity (dot product) between two vectors.
    It then cubes the dot product (as in MINERVA2/HyGene).
    
    Parameters
    ----------
    inputVector1 : vector
        Item or context vector of one dimension.
    inputVector2 : vector
        Item or context vector of one dimension.

    Returns
    ----------
    Float object of similarity calculation.
    """
    
    S = 0
    numFeats = len(inputVector2)
    for i in range(0,len(inputVector2)):
        S = S + (inputVector1[i]*inputVector2[i]) 
        if inputVector1[i] == 0 and inputVector2[i]== 0:
            numFeats = numFeats - 1
    return np.power(S/numFeats,3)


def Icompute(Memory,TestList):

    """
    Computes the nonlinear summed-similarity metric for each test item. 
    
    Parameters
    ----------
    Memory : matrix
        Matrix of vectors representing the studied item-context stimuli degraded
        by the learning parameter (L).
    TestList : matrix
        Matrix of vectors representing the old + new test stimuli.

    Returns
    ----------
    similarity : matrix
        A matrix of the summed-similarity estimations for each test item by
        each study item (dimensions: test list length x study list length).
    similarity2 : vector
        A vector representing the column sums of the similarity matrix.
    
    """
    
    similarity = np.zeros((len(TestList),len(Memory)))
    for i in range(len(TestList)):
        for j in range(len(Memory)):
            similarity[i,j] = vectorSimilarity(Memory[j],TestList[i],)
    similarity2 = np.sum(similarity,axis=1)
    return similarity,similarity2


def recall_contexts(memory, test, A, NumContexts):
    
    """
    Attempts to recall the original contexts based on the degraded memory item
    vectors ("memory") given the test list.
    
    Parameters
    ----------
    memory : matrix
        Matrix of vectors representing the studied item-context stimuli degraded
        by the learning parameter (L).
    test : matrix
        Matrix of vectors representing the old + new test stimuli.
    A : matrix
        A matrix of the summed-similarity estimations for each test item by
        each study item (dimensions: test list length x study list length).
    NumContexts : integer
        Integer specifying the number of contexts (default - 2).

    Returns
    ----------
    C : matrix
        Matrix of information related to context retrieval.
    """
    
    C = np.zeros((np.shape(test)[0],np.shape(memory)[1]))
    for testitem in range(np.shape(test)[0]):
        Cj = np.zeros((np.shape(memory)[0],np.shape(memory)[1]))
        for memtrace in range(np.shape(memory)[0]):
            for feature in range(np.shape(memory)[1]):
                Cj[memtrace,feature] = A[testitem,memtrace] * memory[memtrace,feature]
        for feature in range(np.shape(memory)[1]):
            for memtrace in range(np.shape(memory)[0]):
                C[testitem,feature] = C[testitem,feature] + Cj[memtrace,feature]
    return C
                
def choose_contexts(C, contexts, NumContexts, Cnodes):
    
    """
    Context production - determines and reports which of the previous contexts
    is most similar to the vector yeilded by retrieval process. It produces a
    matrix of size TestItems x Cnodes of the summed dot products of the context 
    traces. The highest summed dot product is the model's best guess at the
    correct context.
    
    Parameters
    ----------
    C : matrix
        MAtrix of information related to context retrieval - derived from 
        "recall_contexts" operation.
    contexts: matrix
        Contains the original context vectors (default - 2 vectors).
    NumContexts: integer
        The number of contexts (default - 2).
    Cnodes : integer
        The number of context nodes in the model.
        
    Returns
    ----------
    DP : matrix
        A matrix of summed dot products resulting from the context algorithm.
    S : vector
        Vector containing the best guess for the context for each item (default -
        0 or 1).
    """
    
    DP = np.zeros((np.shape(C)[0],NumContexts))
    S = np.zeros(np.shape(C)[0])
    for testitem in range(np.shape(C)[0]):
        for context in range(NumContexts):
            DP[testitem,context] = np.dot(C[testitem,np.shape(C)[1]-Cnodes:np.shape(C)[0]],contexts[context])
    for testitem in range(np.shape(C)[0]):
        S[testitem] = np.argmax(DP[testitem])
    return DP,S

def response(S, familiarity):
    """
    Provides a test repsonse given the trace strength of the items as well as the
    best guess for the context. If the trace is above zero and the guessed context
    is for the to-be-included condition, the response is coded as "1".
    
    Parameters
    ----------
    S : vector
        Vector containing the best guess for the context for each item (default -
        0 or 1).
    familiarity : matrix
        A matrix of the summed-similarity estimations for each test item by
        each study item (dimensions: test list length x study list length) whose
        numbers are above the given criterion.    
    
    Returns
    ----------
    R : vector
        Vector of the model's test responses.
    """
    R = np.zeros(len(familiarity))
    for i in range(len(familiarity)):
        if np.isnan(familiarity[i])==False and S[i]==0:
            R[i] = 1
        else:
            R[i] = 0
    return R


def compute_parameters(F, R, NumStudyItems, NumDistract, indexContextOne, indexContextTwo):
    
    """
    Computes performance metrics.
    
    Parameters
    ----------
    F : matrix
        A matrix of the summed-similarity estimations for each test item by
        each study item (dimensions: test list length x study list length) whose
        numbers are above the given criterion.     
    R : vector
        Vector of the model's test responses.
    NumStudyItems : integer
        The number of items at study.
    NumDistract : integer
        Number of new distractors on the exclusion test.
    indexContextOne : vector
        Lists all studied objects that were in Context One.
    indexContextTwo : vector
        Lists all studied objects that were in Context Two.
        
    Returns
    ----------
    HR : float
        Hit Rate
    FARtbx : float
        False alarm rate for the to-be-excluded items
    FARnew : float
        False alarm rate for new items.
    FARnew_F : float
        False alarm rate for new items on the basis of familiarity.
    """

    numTBE = len(indexContextOne)
    numTBX = len(indexContextTwo)
    
    HR = sum(R[indexContextOne])/numTBE
    
    FARtbx = sum(R[indexContextTwo])/numTBX
    
    FARnew = sum(R[NumStudyItems:,])/NumDistract
    
    FARnew_F = np.nansum(F[NumStudyItems:,])/NumDistract
    
    return HR,FARtbx,FARnew,FARnew_F
    

    
