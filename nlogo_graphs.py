'''
A module to safely report graph structures and functions
from Python to NetLogo.

Author: Nick Rabb (nick.rabb2@gmail.com)
'''

import networkx as nx
import numpy as np
import mag
import community as community_louvain
from messaging import *
from random import random
from kronecker import kronecker_pow
from utils import find_nearest, get_keys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter

'''
Return a NetLogo-safe Erdos-Renyi graph from the NetworkX package.

:param n: The number of nodes for the graph.
:param p: The probability of two random nodes connecting.
'''
def ER_graph_bidirected(n, p):
  G = nx.erdos_renyi_graph(n, p)
  return nlogo_safe_nodes_edges(bidirected_graph(G))

'''
Return a NetLogo-safe Erdos-Renyi graph from the NetworkX package.

:param n: The number of nodes for the graph.
:param p: The probability of two random nodes connecting.
'''
def ER_graph(n, p):
  G = nx.erdos_renyi_graph(n, p)
  return nlogo_safe_nodes_edges(G)

'''
Return a Netlogo-safe Watts-Strogatz graph from the NetworkX package.

:param n: The number of nodes.
:param k: The number of initial neighbors.
:param p: The probability of an edge rewiring.
'''
def WS_graph_bidirected(n, k, p):
  G = nx.watts_strogatz_graph(n, k, p)
  return nlogo_safe_nodes_edges(bidirected_graph(G))

'''
Return a Netlogo-safe Watts-Strogatz graph from the NetworkX package.

:param n: The number of nodes.
:param k: The number of initial neighbors.
:param p: The probability of an edge rewiring.
'''
def WS_graph(n, k, p):
  G = nx.watts_strogatz_graph(n, k, p)
  return nlogo_safe_nodes_edges(G)

'''
Return a Netlogo-safe Barabasi-Albert graph from the NetworkX package.

:param n: The number of nodes.
:param m: The number of edges to connect with when a node is added.
'''
def BA_graph_bidirected(n, m):
  G = nx.barabasi_albert_graph(n, m)
  return nlogo_safe_nodes_edges(bidirected_graph(G))

'''
Return a Netlogo-safe Barabasi-Albert graph from the NetworkX package.

:param n: The number of nodes.
:param m: The number of edges to connect with when a node is added.
'''
def BA_graph(n, m):
  G = nx.barabasi_albert_graph(n, m)
  return nlogo_safe_nodes_edges(G)

'''
Create a MAG graph for N nodes, given L attributes, and a style of connection
if there is no specified connection affinity matrix.

:param n: The number of nodes.
:param attrs: A list of attributes to gather Theta affinity matrices for in order
to properly calculate the product of all attribute affinities for the matrix.
:param style: A string denoting how to connect the attributes - default, homophilic, or heterophilic.
'''
def MAG_graph_bidirected(n, attrs, style, resolution):
  (p_edge, L) = mag.attr_mag(n, attrs, style, resolution)
  # print(p_edge)
  # print(L)
  G = nx.Graph()
  G.add_nodes_from(range(0, len(p_edge[0])))
  for i in range(0,len(p_edge)):
    for j in range(0,len(p_edge)):
      rand = random()
      if (rand <= p_edge[(i,j)]):
        # if (abs(L[i][0]-L[j][0]) >= 2):
          # print(f'Chance to connect {L[i]} and {L[j]}: {p_edge[(i,j)]}')
          # print(f'Rolled {rand}: {rand <= p_edge[(i,j)]}')
        G.add_edge(i, j)
  # print(f'Num edges: {len(G.edges)}')
  nlogo_G = nlogo_safe_nodes_edges(bidirected_graph(G))
  nlogo_G.update({'L': L})
  return nlogo_G

'''
Create a MAG graph for N nodes, given L attributes, and a style of connection
if there is no specified connection affinity matrix.

:param n: The number of nodes.
:param attrs: A list of attributes to gather Theta affinity matrices for in order
to properly calculate the product of all attribute affinities for the matrix.
:param style: A string denoting how to connect the attributes - default, homophilic, or heterophilic.
'''
def MAG_graph(n, attrs, style, resolution):
  (p_edge, L) = mag.attr_mag(n, attrs, style, resolution)
  # print(p_edge)
  # print(L)
  G = nx.Graph()
  G.add_nodes_from(range(0, len(p_edge[0])))
  for i in range(0,len(p_edge)):
    for j in range(0,len(p_edge)):
      rand = random()
      if (rand <= p_edge[(i,j)]):
        # if (abs(L[i][0]-L[j][0]) >= 2):
          # print(f'Chance to connect {L[i]} and {L[j]}: {p_edge[(i,j)]}')
          # print(f'Rolled {rand}: {rand <= p_edge[(i,j)]}')
        G.add_edge(i, j)
  # print(f'Num edges: {len(G.edges)}')
  nlogo_G = nlogo_safe_nodes_edges(G)
  nlogo_G.update({'L': L})
  return nlogo_G

def kronecker_graph(seed, k):
  '''
  Make a kronecker graph from a given seed to a power.

  :param seed: An np array to Kronecker power.
  :param k: An integer to raise the graph to the Kronecker power of.
  '''
  G_array = kronecker_pow(seed, k)
  G = nx.Graph()
  G.add_nodes_from(range(0, G_array.shape[0]))
  for i in range(G_array.shape[0]):
    row = G_array[i]
    for j in range(G_array.shape[1]):
      if i == j:
        continue
      p = row[j]
      if random() < p:
        G.add_edge(i,j)
  largest_connected_component = max(nx.connected_components(G), key=len)
  G.remove_nodes_from(G.nodes - largest_connected_component)
  # return G
  return nlogo_safe_nodes_edges(G)

def kronecker_graph_bidirected(seed, k):
  '''
  Make a kronecker graph from a given seed to a power.

  :param seed: An np array to Kronecker power.
  :param k: An integer to raise the graph to the Kronecker power of.
  '''
  G_array = kronecker_pow(seed, k)
  G = nx.Graph()
  G.add_nodes_from(range(0, G_array.shape[0]))
  for i in range(G_array.shape[0]):
    row = G_array[i]
    for j in range(G_array.shape[1]):
      if i == j:
        continue
      p = row[j]
      if random() < p:
        G.add_edge(i,j)
  largest_connected_component = max(nx.connected_components(G), key=len)
  G.remove_nodes_from(G.nodes - largest_connected_component)
  return nlogo_safe_nodes_edges(bidirected_graph(G))

def bidirected_graph(G):
  '''
  Convert an undirected graph to a directed graph where each
  undirected edge becomes two directed edges.

  :param G: An undirected networkx graph.
  '''
  bidirected_G = nx.DiGraph()
  for edge in G.edges:
    bidirected_G.add_edge(edge[0], edge[1])
    bidirected_G.add_edge(edge[1], edge[0])
  return bidirected_G

'''
Return NetLogo-safe graph structures.

:param G: The networkx graph to convert.
'''
def nlogo_safe_nodes_edges(G):
  nodes = list(G.nodes)
  edges = [ [e[0], e[1]] for e in G.edges ]
  return { 'nodes': nodes, 'edges': edges }

'''
Convert a graph from NetLogo to a networkx graph.

:param citizens: A list of citizen agents' brain objects.
:param friend_links: A list of citizen agents' friend links
'''
def nlogo_graph_to_nx(citizens, friend_links):
  G = nx.Graph()
  for cit in citizens:
    cit_id = int(cit['ID'])
    G.add_node(cit_id)
    for attr in cit['malleable']:
      G.nodes[cit_id][attr] = cit[attr]
    for attr in cit['prior']:
      G.nodes[cit_id][attr] = cit[attr]
  for link in friend_links:
    link_split = link.split(' ')
    end1 = link_split[1]
    end2 = link_split[2].replace(')','')
    G.add_edge(int(end1), int(end2))
  return G

def nlogo_graph_to_nx_with_media(citizens, friend_links, media, subscribers):
  G = nx.Graph()
  # agents = citizens + media
  links = friend_links + subscribers
  for agent in citizens:
    cit_id = int(agent['ID'])
    G.add_node(cit_id)
    G.nodes[cit_id]['type'] = 'citizen'
    for attr in agent['malleable']:
      G.nodes[cit_id][attr] = agent[attr]
    for attr in agent['prior']:
      G.nodes[cit_id][attr] = agent[attr]
  for agent in media:
    media_id = int(agent['ID'])
    G.add_node(media_id)
    G.nodes[media_id]['type'] = 'media'
    for attr in agent['malleable']:
      G.nodes[media_id][attr] = agent[attr]
    for attr in agent['prior']:
      G.nodes[media_id][attr] = agent[attr]
  for link in links:
    link_split = link.split(' ')
    end1 = link_split[1]
    end2 = link_split[2].replace(')','')
    G.add_edge(int(end1), int(end2))
  return G

def influencer_paths(G, subscribers, target):
  target_id = int(target.split(' ')[1].replace(')', ''))
  return { subscriber.split(' ')[2].replace(')',''): nx.all_simple_paths(G, subscriber.split(' ')[2].replace(')',''), target, cutoff=5) for subscriber in subscribers }

'''
Get all paths from an influencer to a target node who only contain nodes within
a certain threshold distance from a given message.

:param citizens: A list of citizen agents' brain objects.
:param friend_links: A list of citizen agents' friend links
:param subscribers: A list of subscribers of the influencer.
:param target: The target node to find paths to.
:param message: The message to use for agent distance.
:param threshold: A value that the distance between message and agent cannot
exceed in valid paths.
'''
def influencer_paths_within_distance(citizens, friend_links, subscribers, target, message, threshold):
  G = nlogo_graph_to_nx(citizens, friend_links)

  # Assign edge weights of the message distance to the first agent in the link
  for e in G.edges:
    G[e[0]][e[1]]['weight'] = dist_to_agent_brain(G.nodes[e[0]], message)

  target_id = int(target.split(' ')[1].replace(')', ''))
  paths = { int(subscriber.split(' ')[2].replace(')','')): nx.dijkstra_path(G, int(subscriber.split(' ')[2].replace(')','')), target_id) for subscriber in subscribers }

  distance_paths = {}
  threshold_paths = {}
  for subscriber in paths.keys():
    dist_path = [ dist_to_agent_brain(G.nodes[v], message) for v in paths[subscriber] ]
    dist_path = dist_path[:-1]
    distance_paths[subscriber] = dist_path
    if sum((np.array(dist_path)-threshold) > 0) == 0:
      threshold_paths[subscriber] = dist_path
      # threshold_paths[subscriber] = paths[subscriber]
  return threshold_paths

def plot_graph_communities(G, level):
  '''

  '''
  dendrogram = community.generate_dendrogram(G)
  partition = community.partition_at_level(dendrogram, level)
  pos = nx.spring_layout(G)
  cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
  nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
  nx.draw_networkx_edges(G, pos, alpha=0.5)
  plt.show()

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

def flint_community(communities, n):   
    list_mean = []
    for level in range(len(communities)):
      partition = communities[level]
      arr = np.array(list(Counter(partition.values()).values()))
      mean = np.mean(arr)
      list_mean.append(mean)
    closest_mean = find_nearest(list_mean, n)
    lvl = list_mean.index(closest_mean)
    partition = communities[lvl]
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

def nlogo_community_sizes_by_level(citizens, social_friends):
  G = nlogo_graph_to_nx(citizens, social_friends)
  dendrogram = community_louvain.generate_dendrogram(G)
  levels = len(dendrogram)
  return [ max(list(community_louvain.partition_at_level(dendrogram, level).values())) for level in range(levels) ]

def nlogo_communities_by_level(citizens, social_friends):
  G = nlogo_graph_to_nx(citizens, social_friends)
  dendrogram = community_louvain.generate_dendrogram(G)
  levels = len(dendrogram)
  return [ community_louvain.partition_at_level(dendrogram, level) for level in range(levels) ]

def media_peer_connections(G):
  media_degrees = np.array([ G.degree(node) for node in G.nodes if G.nodes[node]['type'] == 'media'])
  num_media = len(media_degrees)
  max_degree = max(media_degrees)
  connection_prob = media_degrees / max_degree
  prob_matrix = np.ones((num_media,num_media)) * connection_prob
  prob_matrix = np.multiply(prob_matrix, 1 - np.identity(num_media))
  rolls = np.random.rand(num_media,num_media)
  connections = (rolls <= prob_matrix).astype(int)
  return connections

def graph_homophily(G):
  '''
  Takes a measure of homophily in the graph based on first-level neighbor
  distance on a given node attribute. Details can be found in Rabb et al. 2022
  Eq (9).

  :param G: The networkx graph to take the measure on.
  '''
  distances = []
  attrs = np.array([ list(G.nodes[node].values()) for node in G.nodes ])
  adj = nx.adj_matrix(G)
  for node in G.nodes:
    norm_vector = np.array([ np.linalg.norm(attr - attrs[node]) for attr in attrs ])
    # Note: adj[node] * norm_vector sums the values already
    distances.append((adj[node] * norm_vector)[0] / adj[node].sum())
  return (np.array(distances).mean(), np.array(distances).var())

def node_degree_centrality(G, node):
  '''
  Get the degree centrality for a single node in G.

  :param G: The graph.
  :param node: The integer node index to get.
  '''
  return nx.degree_centrality(G)[node]

def nodes_degree_centrality(G, nodes):
  return sum([ node_degree_centrality(G, node) for node in nodes ])

def test_ws_graph_normal(n, k, p):
  G = nx.watts_strogatz_graph(n, k, p)
  agent_bels = normal_dist_multiple(7, 3, 1, n, 2)
  for i in range(n):
    G.nodes[i]['A'] = agent_bels[i][0]
    G.nodes[i]['B'] = agent_bels[i][1]
  return G
