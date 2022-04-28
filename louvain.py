import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from networkx import community as community_louvain
from collections import Counter
import numpy as np
import math

from nlogo_graphs import nlogo_graph_to_nx

"""
Find element in the array with smallest distance from given value
"""
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

"""
Returns list of keys with given value from dictionary

:param dict: dictionary of communities
:param value: value to search for
"""
def get_keys(dict, value):
    list_keys = list()
    list_items = dict.items()
    for item  in list_items:
        if item[1] == value:
            list_keys.append(item[0])
    return list_keys
    
def flint_community_nlogo(citizens, social_friends, n):
    '''
    First convert the citizen and edge arrays into a graph, then
    call the flint community function.

    :param citizens: An array of citizen nodes in the graph.
    :param social_friends: An array of citizen social friend edges in the graph.
    :param n: The ideal size of community to look for.
    '''
    G = nlogo_graph_to_nx(citizens, social_friends)
    return flint_community(G, n)

"""
Returns an array of the nodes in the community of a graph with 
number of nodes closest to n

:param G: graph to detect communities
:param n: approximate number of nodes in community    
"""

def flint_community(G, n):   
  def flint_community(G, n):  
    # partition = community_louvain.best_partition(G)
    dendo = community_louvain.generate_dendrogram(G)
    list_mean = []
    for level in range(len(dendo) - 1):
        partition = community_louvain.partition_at_level(dendo, level)
        arr = np.array(list(partition.values()))
        mean = np.mean(arr)
        list_mean.append(mean)
    closest_mean = find_nearest(list_mean, n)
    lvl = list_mean.index(closest_mean)
    partition = community_louvain.partition_at_level(dendo, lvl)
    # value is number of nodes in community [key]
    count = Counter(partition.values())
    # array of numbers of nodes in each community
    values = list(count.values())
    # the number of nodes in the community that has the closest size to n
    comm_size = find_nearest(values, n)
    # community that corresponds to closest number of nodes
    closest_community = list(count.keys())[list(count.values()).index(comm_size)]    
    nodes_in_partition = get_keys(partition, closest_community)
    return nodes_in_partition
