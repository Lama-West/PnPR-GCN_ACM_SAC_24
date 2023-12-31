"""
This script stores the utility functions for this repository.
"""

import numpy as np
import pandas as pd
import networkx as nx
import itertools

def ReadCSV(path = "AL_CPL_Originial_Data/geometry.preqs"):
    """
    Simple function that reads the AL-CPL data and structures the dataframe. It also removes the underscores from the names of concepts.
    """
    csv = pd.read_csv(path, names=["Concept", "Prerequisite"])

    ## Concepts are stored with '_' in their names, let's undo that
    csv.Concept      = csv.Concept.apply(lambda x: x.replace('_',' '))
    csv.Prerequisite = csv.Prerequisite.apply(lambda x: x.replace('_',' '))

    return csv

def MergingPreqsAndPairs(csv_preqs, csv_pairs):
    '''Generating the Dataset of labeled concept pairs for a specific 'domain'.
    Basically, we read and merge the pairs stored in the files `.preqs` and `.pairs` following the guideline of the AL-CPL dataset.
    https://github.com/harrylclc/AL-CPL-dataset
    '''
    csv_preqs['label_prereq'] = np.ones_like(csv_preqs.Concept)
    csv_pairs['label_prereq'] = np.zeros_like(csv_pairs.Concept)
    csv_preqs['relations_1']  = csv_preqs.apply(lambda x: (x.Concept, x.Prerequisite), axis=1)
    csv_pairs['relations_1']  = csv_pairs.apply(lambda x: (x.Concept, x.Prerequisite), axis=1)

    ## Concatenates 0s and 1s relations
    ds = pd.concat([csv_preqs, csv_pairs[~csv_pairs.relations_1.isin(csv_preqs.relations_1)]]).drop(columns='relations_1')

    return ds

## --------------------- Functions useful to infer prerequisite relations by transitivity given a directed graph --------------------- ##
def cartesianProductOfLists(list1, list2):
      """Given two lists, this functions returns the cartesian product of this two lists.
         Ex: [1,2], [3,4,5] -> [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]

      Args:
          list1 (list): Input list number one.
          list2 (lsit): Input list number two.

      Returns:
          list of tuples: The cartesian product of both inputted lists.
      """
      return list(itertools.product(list1, list2))

def LinkedPrerequisitePairsFromRoot(root, DG):
    '''
    This function computes all the possible pairs we can infer by transitivity given the directed graph `DG`
    and the root of one of its connected component `root`. In other terms, it computes the transitive closure 
    of the directed graph `DG`.

    The algorithm goes as follows:
        For each root node, we compute all the possible pairs between itself and its descendants.
        Then, we compute all possible pairs between the descendants and their own descendants.
    '''
    successors = list(nx.descendants(DG, root))
    extracted_pairs = cartesianProductOfLists([root], successors)

    for succ in successors:
        successors_2 = list(nx.descendants(DG, succ))
        extracted_pairs += cartesianProductOfLists([succ], successors_2)
    
    return extracted_pairs

def AllLinkedPrerequisitePairs(DG):
    '''
    This function extends LinkedConceptPairs. First, it computes all the roots of the connected components of the directed graph `DG`.
    Then, it performs LinkedConceptPairs on all of the roots.
    '''
    # Roots have an in degree of 0 in directed graphs
    # roots = [n for n,d in DG.in_degree() if d==0] 
    # list_of_prerequisite_pairs = []

    # for root in roots:
    #     ## Generating all possible prerequisite relations:    
    #     list_of_prerequisite_pairs += LinkedPrerequisitePairsFromRoot(root, DG)
        
    # return list_of_prerequisite_pairs
    return list(nx.transitive_closure_dag(DG).edges)

def PercentageOfInferablePrerequisites(train_G, test_df):
    '''This function returns the percentage of edges in the test set `test_df` that are inferable by transitivity from edges in the training set.
       Here, `train_G` denotes the graph associated with the training set.

    Args:
        train_G (networkx.classes.digraph.DiGraph): The graph associated with the training set.
        test_df (pandas.core.frame.DataFrame): The testing set dataframe.
    '''
    test_df              = test_df.copy(deep=True)[test_df.label_prereq==1]
    list_of_inferable_prerequisite_pairs = AllLinkedPrerequisitePairs(train_G)
    test_df['relations'] = test_df.apply(lambda x: (x.Prerequisite, x.Concept), axis=1)

    return test_df.relations.isin(list_of_inferable_prerequisite_pairs).sum()/test_df.label_prereq.sum()*100

## --------------------- END --------------------- ##

## Function that computes the percentage of pairs labeled 0 in the dataframe df
ZeroPercentageInDF = lambda df: df.label_prereq.value_counts()[0]/df.shape[0]*100

## Function that computes the number of unique concepts in the dataframe df
NumberOfConceptsInDF = lambda df: len(pd.concat([df.Concept, df.Prerequisite]).unique())

def RatioOfNonTransitiveEdges(df, PrereqG, verbose=False):
    '''Computes and returns the ration of Non Transitive Edges in the Pandas Data Frame `df`.

    Args:
        df (pandas.core.frame.DataFrame): A Pandas Data Frame containing the data.
        PrereqG (networkx.classes.digraph.DiGraph): The Directed Prerequisite Graph without transitive relations.
        verbose (bool, optional): A boolean that indicates if the user wishes to print out the ratio. Defaults to False.
    '''
    df              = df.copy(deep=True)
    df['relations'] = df.apply(lambda x: (x.Prerequisite, x.Concept), axis=1)
    ratio_of_non_transitive_edges = df.relations.isin(nx.edges(PrereqG)).sum()/df.label_prereq.sum()
    if(verbose):
        print("Percentage of non transitive edges : {0:.2f}%".format(ratio_of_non_transitive_edges*100))

    return ratio_of_non_transitive_edges

def AddingStatsToDict(info_split:dict, split_df, DG, train_G, Split_Type:str, Fold_number:int) -> dict:
    ## Primery Key of Dataset
    info_split['Split_Type'].append(Split_Type)
    info_split['Fold_number'].append(Fold_number)

    ## Number of Unique Concepts in Split
    info_split['Number_of_unique_concepts'].append(NumberOfConceptsInDF(split_df))

    ## Percentage of 0s in split
    info_split['Percentage_of_NonPrerequisite_Relations'].append(ZeroPercentageInDF(split_df))

    ## Number of non-Transitive Induced Relations in Split
    info_split['Percentage_of_non_transitive_edges'].append(RatioOfNonTransitiveEdges(split_df, DG, verbose=False)*100)

    ## Percentage of Edges Inferable by Transitivity from the Train split in Test Split
    if(Split_Type.lower() in 'test'):
        info_split['Percentage_of_Edges_Inferable_by_Transitivity_in_Test_Split'].append(PercentageOfInferablePrerequisites(train_G, split_df))
    else:
        info_split['Percentage_of_Edges_Inferable_by_Transitivity_in_Test_Split'].append(np.nan)

    
    return info_split