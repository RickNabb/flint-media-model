import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import numpy as np
import math

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
    
"""
Returns an array of the nodes in the community of a graph with 
number of nodes closest to n

:param G: graph to detect communities
:param n: approximate number of nodes in community    
"""
def flint_community(G, n):   
    partition = community_louvain.best_partition(G)
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
