"""
This Script implements the in-domain 5 fold cross-validation on AL-CPL as described in previous papers 
[https://clgiles.ist.psu.edu/pubs/eaai18.pdf, https://aclanthology.org/W19-4430/, https://www.semanticscholar.org/reader/87d51b4b8459a2bfbca2f489d4b75987c634449b]
"""

import numpy as np
import pandas as pd
import networkx as nx
import os

from sklearn.model_selection import KFold
from Scripts.utils import ReadCSV, MergingPreqsAndPairs, AddingStatsToDict

if __name__ == "__main__":
    graphs_dir = 'AL_CPL_Concept_Graphs_Without_Transitivity/'
    data_dir   = 'AL_CPL_Originial_Data/'
    split_dir  = 'Random_Split/'

    domains = ["data_mining", "geometry", "physics", "precalculus"]
    concepts_preq = {}
    concepts_all  = {}

    ## Loading all Graphs
    for domain in domains:
        ## Creating the Directory if needed
        domain_split_dir = split_dir + '{0}/'.format(domain)
        if(not os.path.exists(domain_split_dir)):
            os.makedirs(domain_split_dir)

        ## Reading and Storing concept pairs
        csv_preqs = ReadCSV(data_dir + "{0}.preqs".format(domain))#Only ones
        csv_pairs = ReadCSV(data_dir + "{0}.pairs".format(domain))#Ones and Zeros

        ## Assempling Dataset with pairs labeled 0 for non-prerequisite relations and 1 prerequisite relations between concepts
        df = MergingPreqsAndPairs(csv_preqs, csv_pairs)

        ## Constructing the prerequisite graph with transitivity
        nxG = nx.from_pandas_edgelist(csv_preqs, source='Prerequisite', target='Concept', create_using=nx.DiGraph)
        # Prerequisite graph without transitivity
        nxG = nx.transitive_reduction(nxG)

        # We spotted a pair labeled wrong in the AL-CPL dataset, so we correct it:
        df.loc[df[(df.Concept == "Symmetry") & (df.Prerequisite =="Geometry")].index, 'label_prereq'] = 1 
        
        # Initialize 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        ## Storing stats on different splits
        info_split = {'Split_Type': [], 'Fold_number': [], 'Number_of_unique_concepts': [], 'Percentage_of_non_transitive_edges': []}
        info_split = {**info_split, **{'Percentage_of_NonPrerequisite_Relations': [], 'Percentage_of_Edges_Inferable_by_Transitivity_in_Test_Split': []}}
        # Adding stats on the entire dataset for the domain
        info_split = AddingStatsToDict(info_split, df, nxG, df, Split_Type='ALL', Fold_number=-1)

        ## Splitting the DataFrame indexes into 5 train/test splits
        for i, _ in enumerate(kf.split(df)):
            ## Splitting the data according to cross validation folds
            train_index, test_index = _
            train_df = df.iloc[train_index]
            test_df  = df.iloc[test_index]

            ## Creating the prerequisite graph without transitivity for train_df
            train_G = nx.from_pandas_edgelist(train_df[train_df.label_prereq==1], source='Prerequisite', target='Concept', create_using=nx.DiGraph)
            train_G = nx.intersection(train_G, nxG)

            ## Storing Statistics
            info_split = AddingStatsToDict(info_split, train_df, nxG, train_G, Split_Type='Train', Fold_number=i+1)
            info_split = AddingStatsToDict(info_split, test_df, nxG, train_G, Split_Type='Test', Fold_number=i+1)

            ## Writing the Splits to Disk
            train_df.to_csv(domain_split_dir + '{0}_train_split_{1}.csv'.format(domain, i+1), index=False)
            test_df.to_csv(domain_split_dir + '{0}_test_split_{1}.csv'.format(domain, i+1), index=False)

        ## Writing the stats to Disk
        info_split = pd.DataFrame(info_split)
        info_split.to_csv(domain_split_dir + '{0}_split_statistics.csv'.format(domain), index=False)
    
    ## Announcing to the user that the data has been generated successfully!
    print("The in-domain 5 fold cross-validation splits have been successfully generated in directory {0}, yay!!".format(split_dir))