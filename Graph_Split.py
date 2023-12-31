"""
This Script implements the in-domain 5 fold cross-validation on AL-CPL as described in previous papers 
[https://clgiles.ist.psu.edu/pubs/eaai18.pdf, https://aclanthology.org/W19-4430/, https://www.semanticscholar.org/reader/87d51b4b8459a2bfbca2f489d4b75987c634449b]
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import random

from sklearn.model_selection import KFold
from Scripts.utils import ReadCSV, MergingPreqsAndPairs, AddingStatsToDict, RatioOfNonTransitiveEdges, AllLinkedPrerequisitePairs


def remove_random_edge(graph):
    '''
    Delete a random edge.
    :param graph: networkx graph
    :return: networkx graph and the removed edge
    '''
    graph = graph.copy()
    edges = list(graph.edges)

    # random edge choice
    chosen_edge = random.choice(edges)

    # remove chosen edge from graph
    graph.remove_edge(chosen_edge[0], chosen_edge[1])

    return graph, chosen_edge

def RemoveNonTransitiveEdges(df, PrereqG):
    df              = df.copy(deep=True)
    df['relations'] = df.apply(lambda x: (x.Prerequisite, x.Concept), axis=1)
    return df.drop(index = df[df.relations.isin(nx.edges(PrereqG))].index).drop(columns='relations')

def RemoveInfereableEdges(df, PrereqG):
    df              = df.copy(deep=True)
    df['relations'] = df.apply(lambda x: (x.Prerequisite, x.Concept), axis=1)
    return df.drop(index = df[df.relations.isin(AllLinkedPrerequisitePairs(PrereqG))].index).drop(columns='relations')

def GraphSplitAlgorithm(df, DG, split_train=.8, verbose=False):
    """
    This function implement the graph split algorithm described in our paper.
    It takes the entire dataset and directed graph as entry.
    """
    ratio_of_non_transitive_edges = RatioOfNonTransitiveEdges(df, DG, verbose=verbose)
    nbre_direct_test = int(np.ceil((1-split_train) * df.label_prereq.sum() * ratio_of_non_transitive_edges))
    nbre_trans_test  = int((1-split_train) * df.label_prereq.sum() - nbre_direct_test)

    train_ds = df.copy()
    test_ds  = pd.DataFrame.from_dict({train_ds.columns[0]:[], train_ds.columns[1]:[]})
    trans_rel  = pd.DataFrame()
    direct_rel = pd.DataFrame()
    PrereqG_copy = DG.copy()

    # Identification and seperation of non transitive relations and transitive relations:
    while(direct_rel.shape[0] < nbre_direct_test):
        # Choose an edge to be removed:
        PrereqG_copy, removed_edge = remove_random_edge(PrereqG_copy)

        # Get the predecessors and the successors of nodes in removed edge
        anc = list(nx.ancestors(DG, removed_edge[0]))
        dec = list(nx.descendants(DG, removed_edge[1]))

        # Select relations that can be deduced by transitivity
        transitive_ds   = train_ds[(train_ds.Concept.isin(dec) & train_ds.Prerequisite.isin(anc) & train_ds.label_prereq==1)]
        transitive_ds_2 = train_ds[(train_ds.Concept.isin([removed_edge[1]]) & train_ds.Prerequisite.isin(anc) & train_ds.label_prereq==1)]
        transitive_ds_3 = train_ds[(train_ds.Concept.isin(dec) & train_ds.Prerequisite.isin([removed_edge[0]]) & train_ds.label_prereq==1)]
        
        # Select Direct Relations (non transitive)
        direct_ds       = train_ds[(train_ds.Concept.isin([removed_edge[1]]) & train_ds.Prerequisite.isin([removed_edge[0]]) & train_ds.label_prereq==1)]

        # Concat relations induced by transitivity
        trans_rel  = pd.concat([trans_rel, transitive_ds, transitive_ds_2, transitive_ds_3]).drop_duplicates(['Concept', 'Prerequisite'])

        # Concat relations that are not induced by transitivity
        direct_rel = pd.concat([direct_rel, direct_ds])

        # Remove Direct Edges in Transitive (See example of angle and geometry to understand why)
        trans_rel = RemoveNonTransitiveEdges(trans_rel, DG)

    if(verbose):
        print("Percentage of non transitive edges before random selection ", direct_rel.shape[0] / (direct_rel.shape[0] + trans_rel.shape[0])*100)
        print("Transitive induced relations we computed - the number of transitive induced relations needed:", trans_rel.shape[0]-nbre_trans_test)

    # Sampling the direct and transitive relations
    trans_rel  = RemoveInfereableEdges(trans_rel, PrereqG_copy)
    trans_rel_in_test  = trans_rel.copy().sample(n=min(nbre_trans_test, trans_rel.shape[0]), replace=False)
    direct_rel_in_test = direct_rel.copy().sample(n=nbre_direct_test, replace=False)
    test_ds = pd.concat([direct_rel_in_test, trans_rel_in_test]).drop_duplicates(['Concept', 'Prerequisite'])

    ## Adding 0 to the test set
    nbre_0_ajouter = int(df.shape[0] - split_train * df.shape[0] - test_ds.shape[0])
    selected_ds_0 = train_ds[train_ds.label_prereq == 0].sample(n=nbre_0_ajouter, replace=False)
    test_ds = pd.concat([test_ds, selected_ds_0])

    ## Removing testing set values from training set
    train_ds = train_ds.merge(test_ds, on=['Concept', 'Prerequisite', 'label_prereq'], how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    return train_ds, test_ds, PrereqG_copy

if __name__ == "__main__":
    graphs_dir = 'AL_CPL_Concept_Graphs_Without_Transitivity/'
    data_dir   = 'AL_CPL_Originial_Data/'
    split_dir  = 'Graph_Split/'

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

        ## Constructing the prerequisite graph with transitivity
        nxG = nx.from_pandas_edgelist(csv_preqs, source='Prerequisite', target='Concept', create_using=nx.DiGraph)
        # Prerequisite graph without transitivity
        nxG = nx.transitive_reduction(nxG)

        ## Assempling Dataset with pairs labeled 0 for non-prerequisite relations and 1 prerequisite relations between concepts
        df = MergingPreqsAndPairs(csv_preqs, csv_pairs)

        # We spotted a pair labeled wrong in the AL-CPL dataset, so we correct it:
        df.loc[df[(df.Concept == "Symmetry") & (df.Prerequisite =="Geometry")].index, 'label_prereq'] = 1 
        
        ## Storing stats on different splits
        info_split = {'Split_Type': [], 'Fold_number': [], 'Number_of_unique_concepts': [], 'Percentage_of_non_transitive_edges': []}
        info_split = {**info_split, **{'Percentage_of_NonPrerequisite_Relations': [], 'Percentage_of_Edges_Inferable_by_Transitivity_in_Test_Split': []}}
        # Adding stats on the entire dataset for the domain
        info_split = AddingStatsToDict(info_split, df, nxG, df, Split_Type='ALL', Fold_number=-1)

        ## Splitting the DataFrame indexes into 5 train/test splits
        for i in range(5):
            ## Splitting the data according to cross validation folds
            train_df, test_df, train_G = GraphSplitAlgorithm(df, nxG, split_train=.8, verbose=False)

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
    print("The in-domain Graph splits have been successfully generated in directory {0}, yay!!".format(split_dir))