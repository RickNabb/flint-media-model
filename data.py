from email import message
from enum import Enum
from random import *
from utils import *
from statistics import mean, variance, mode
from copy import deepcopy
from plotting import *
from nlogo_colors import *
from nlogo_graphs import read_graph_to_nx_with_media
import itertools
import pandas as pd
import os
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency, truncnorm, percentileofscore
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import json

"""
BELIEF ATTRIBUTES
"""

NUM_BELIEF_BUCKETS = 32

discrete = range(NUM_BELIEF_BUCKETS)

# class TestDiscrete7(Enum):
#   STRONG_DISBELIEF=12
#   DISBELIEF=1
#   MOD_DISBELIEF=2
#   UNCERTAIN=3
#   MOD_BELIEF=4
#   BELIEF=5
#   STRONG_BELIEF=6

class Attributes(Enum):
  A = discrete

def attrs_as_array(attr):
  # return map(lambda a: a.value, list(attr.value))
  return attr.value

"""
STATS STUFF
"""

'''
Return a tuple of summary statistical measures given a list of values.

:param l: The list to calculate stats for.
'''
def summary_statistics(l):
  if len(l) >= 2:
    return (mean(l), variance(l), mode(l))
  elif len(l) >= 1:
    return (mean(l), -1, mode(l))
  else:
    return (-1, -1, -1)

"""
Sample a distribution given a specific attribute. This distribution may
also depend on another, and if so, the function recursively calls
itself to return the needed dependency.

:param attr: An attribute from the Attribues enumeration to sample
its approriate distribution in the empirical data.
"""
def random_dist_sample(attr, resolution, given=None):
    return AttributeValues[attr.name]['vals'](resolution)

"""
Sample an attribute with an equal distribution over the values.

:param attr: The attribute to sample - e.g. Attributes.I.
"""
def random_sample(attr):
  rand = int(math.floor(random() * len(list(attr.value))))
  val = list(attr.value)[rand]
  return val.value

def test_random_sample():
  print(random_sample(Attributes.VG))

"""
RELEVANT EMPIRICAL DATA
"""

# Attribute A distribution values
HomophilicThetaRow = lambda row, l, p, s, d: [ 1/(1 + d + s * abs(pow(row - i, p))) for i in range(0, l) ]
SquareHomophilicThetaRow = lambda row, l: HomophilicThetaRow(row, l, 2, 5, 0.25)
LinearHomophilicThetaRow = lambda row, l: HomophilicThetaRow(row, l, 1, 5, 0.25)

HeterophilicThetaRow = lambda row, l, p, s, d: [ (abs(pow(i-row,p)))/(s*pow(max(abs(l-row-1), row)+d,p)) for i in range(0, l) ]
SquareHeterophilicThetaRow = lambda row, l: HeterophilicThetaRow(row, l, 2, 2, 0)
LinearHeterophilicThetaRow = lambda row, l: HeterophilicThetaRow(row, l, 1, 2, 0)

# AVals = np.ones(NUM_BELIEF_BUCKETS) #[1, 1, 1, 1, 1, 1, 1]
# ADist = create_discrete_dist_sm(AVals)
AMAGDefaultTheta = lambda resolution: np.ones((resolution,resolution)) * 0.05
AMAGHomophilicTheta = lambda resolution: np.matrix([ HomophilicThetaRow(i, resolution, 2, 50, 5) for i in range(0, resolution) ])
AMAGHeterophilicTheta = lambda resolution: np.matrix([ HeterophilicThetaRow(i, resolution, 2, 5, 1) for i in range(0, resolution) ])

def uniform_dist_multiple(maxx, n, k):
  '''
  Return a series of samples drawn from a uniform distribution from [0, max] where each of en samples has k entries.

  :param maxx: The maximum to draw from.
  :param n: The number of k entry samples to draw.
  :param k: The number of entries per n sample.
  '''
  samples = [ uniform_dist(maxx, n) for i in range(k) ]
  return [ [ samples[i][j] for i in range(k) ] for j in range(n) ]

def uniform_dist(maxx, n):
  '''
  Draw n samples from a uniform distribution from [0, maxx]

  :param maxx: The maximum to draw from.
  :param n: The number of samples to take.
  '''
  return np.array(list(map(lambda el: math.floor(el), np.random.uniform(low=0, high = maxx, size=n))))

def normal_dist_multiple(maxx, mean, sigma, n, k):
  '''
  Return a series of samples drawn from a normal distribution from
  [0, max] with mean and std deviation specified, where each of en
  samples has k entries.

  :param maxx: The maximum to draw from.
  :param mean: The mean of the distribution.
  :param sigma: The standard deviation of the distribution.
  :param n: The number of k entry samples to draw.
  :param k: The number of entries per n sample.
  '''
  samples = [ normal_dist(maxx, mean, sigma, n) for i in range(k) ]
  return [ [ samples[i][j] for i in range(k) ] for j in range(n) ]

def normal_dist(maxx, mean, sigma, n):
  '''
  Draw n samples from a truncated normal distribution from [0, maxx]
  with mean and sigma specified.

  :param maxx: The maximum to draw from.
  :param mean: The mean of the distribution.
  :param sigma: The standard deviation of the distribution.
  :param n: The number of samples to take.
  '''
  lower=-0.5
  upper=maxx+0.5
  # mean=math.floor(resolution/2)
  # sigma=mean/3
  dist = truncnorm((lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
  return np.array(list(map(lambda el: round(el), dist.rvs(n))))

AttributeValues = {
  Attributes.A.name: {
    "vals": normal_dist,
    "depends_on": None
  }
}

# AttributeDistributions = {
#   Attributes.A.name: {
#     "dist": ADist,
#     "depends_on": None
#   }
# }

AttributeMAGThetas = {
  Attributes.A.name: {
    'default': AMAGDefaultTheta,
    'homophilic': AMAGHomophilicTheta,
    'heterophilic': AMAGHeterophilicTheta
  }   
}

'''
NETLOGO PARSING
'''

def nlogo_list_to_arr(list_str):
    return [ el.replace('[', '').strip().split(' ') for el in list_str[1:len(list_str)-1].split(']') ]

def nlogo_replace_agents(string, types):
    for type in types:
        string = string.replace(f'({type} ', f'{type}_')
    return string.replace(')','')

'''
Parse a NetLogo mixed dictionary into a Python dictionary. This is a nightmare.
But it works.

:param list_str: The NetLogo dictionary as a string.
'''
def nlogo_mixed_list_to_dict(list_str):
  return nlogo_parse_chunk(list_str)

def nlogo_mixed_list_to_dict_rec(list_str):
  # print(f'processing {list_str}')
  if list_str[0] == '[' and list_str[len(list_str)-1] == ']' and list_str.count('[') == 1:
    return nlogo_parse_chunk(list_str)

  d = {}
  chunk = ''
  stack_count = 0
  for i in range(1, len(list_str)-1):
    chunks = []
    char = list_str[i]
    chunk += char
    if char == '[':
      stack_count += 1
    elif char == ']':
      stack_count -= 1

      if stack_count == 0:
        # print(f'parsing chunk: {chunk}')
        parsed = nlogo_parse_chunk(chunk)
        # print(f'parsed: {parsed}')
        d[list(parsed.keys())[0]] = list(parsed.values())[0]
        chunk = ''
      # chunks[stack_count] += char
  return d

def nlogo_parse_chunk(chunk):
  chunk = chunk.strip().replace('"','')
  if chunk.count('[') > 1 and chunk[0] == '[':
    return nlogo_mixed_list_to_dict_rec(chunk[chunk.index('['):].strip())
  elif chunk.count('[') > 1 or chunk[0] != '[':
    return { chunk[0:chunk.index('[')].strip(): nlogo_mixed_list_to_dict_rec(chunk[chunk.index('['):].strip()) }

  pieces = chunk.strip().replace('[','').replace(']','').split(' ')
  if len(pieces) == 2:
    return { pieces[0]: pieces[1] }
  else:
    return pieces

'''
FILE I/O
'''

DATA_DIR = 'D:/school/grad-school/Tufts/research/flint-media-model'

def save_graph(path, cit, cit_social, media, media_sub, flint_citizens):
    cit_arr = nlogo_list_to_arr(nlogo_replace_agents(cit, [ 'citizen' ]))
    cit_social_arr = nlogo_list_to_arr(nlogo_replace_agents(cit_social, [ 'citizen' ]))
    media_arr = nlogo_list_to_arr(nlogo_replace_agents(media, [ 'media' ]))
    media_sub_arr = nlogo_list_to_arr(nlogo_replace_agents(media_sub, [ 'media', 'citizen' ]))

    f = open(path, 'w')
    f.write(f'CITIZENS {len(cit_arr)-1}\n')
    for c in filter(lambda el: len(el) > 1, cit_arr):
      cit_id = c[0].replace("citizen_","")
      f.write(f'{cit_id},{c[1]},{int(cit_id) in flint_citizens}\n')

    f.write(f'CITIZEN_SOCIAL_LINKS {len(cit_social_arr)-1}\n')
    for c in filter(lambda el: len(el) > 1, cit_social_arr):
        f.write(f'{c[0].replace("citizen_","")},{c[1].replace("citizen_","")}\n')

    f.write(f'MEDIA {len(media_arr)-1}\n')
    for c in filter(lambda el: len(el) > 1, media_arr):
        f.write(f'{c[0].replace("media_","")},{c[1]}\n')

    f.write(f'MEDIA_SUB_LINKS {len(media_sub_arr)-1}\n')
    for c in filter(lambda el: len(el) > 1, media_sub_arr):
        ordered = sorted(c)
        f.write(f'{ordered[0].replace("citizen_","")},{ordered[1].replace("media_","")}\n')

    f.close()

def read_subrange(lines, into):
    for line in lines:
        into.append(line.split(','))

def read_graph(path):
    f = open(path, 'r')
    raw = f.read()
    f.close()
    cit = []
    cit_social = []
    media_arr = []
    media_sub_arr = []
    lines = raw.split('\n')
    i = 0

    while i < len(lines):
      line = lines[i]
      if 'CITIZEN_SOCIAL_LINKS' in line:
        n = int(line.split(' ')[1])
        read_subrange(lines[i+1:i+1+n], cit_social)
        i += n+1
      elif 'CITIZEN' in line:
        n = int(line.split(' ')[1])
        read_subrange(lines[i+1:i+1+n], cit)
        i += n+1
      elif 'MEDIA_SUB_LINKS' in line:
        n = int(line.split(' ')[1])
        read_subrange(lines[i+1:i+1+n], media_sub_arr)
        i += n+1
      elif 'MEDIA' in line:
        n = int(line.split(' ')[1])
        read_subrange(lines[i+1:i+1+n], media_arr)
        i += n+1
      else:
        i += 1
    return (cit, cit_social, media_arr, media_sub_arr)

def write_message_data(path, filename, messages_adopted):
  '''
  '''
  messages_adopted_py = { int(tick): adopter_sender_pair_dict for tick, adopter_sender_pair_dict in messages_adopted.items() }
  with open(f'{path}/{filename}_messages_adopted.json', 'w', encoding='utf-8') as f:
    json.dump(messages_adopted_py, f, ensure_ascii=False)

def write_link_formation_data(path, filename, links_formed):
  '''
  '''
  links_formed_py = { int(tick): organizer_organized_pair_dict for tick, organizer_organized_pair_dict in links_formed.items() }
  with open(f'{path}/{filename}_links_formed.json', 'w', encoding='utf-8') as f:
    json.dump(links_formed_py, f, ensure_ascii=False)

"""
ANALYSIS FUNCTIONS
"""

def process_multiple_sim_data(path):
  for file in os.listdir(path):
    data = process_sim_data(f'{path}/{file}')
    stats = citizen_message_statistics(data[0], data[1])

'''
Parse a NetLogo chart export .csv file. This requires a single chart export file
and should not be run on an entire world export. This will return a dictionary
of plot points in a data frame, keyed by the pen name.

:param path: The path to the chart .csv file.
'''
def process_chart_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  chunks = raw.split('\n\n')

  model_lines = chunks[1].replace('"','').split('\n')
  model_keys = model_lines[1].split(',')
  model_vals = model_lines[2].split(',')
  model_props = { model_keys[i]: model_vals[i] for i in range(len(model_keys)) }

  prop_lines = chunks[2].replace('"','').split('\n')
  chart_props = {}
  chart_props['color'] = {}
  keys = prop_lines[1].split(',')
  vals = prop_lines[2].split(',')
  for i in range(0, len(keys)):
    chart_props[keys[i]] = vals[i]

  data_sets = {}
  chart_lines = chunks[4].split('\n')
  
  data_set_splits = []
  split_line = chart_lines[0].split(',')
  for i in range(0, len(split_line)):
    el = split_line[i].replace('"','')
    if el != '':
      data_set_splits.append((el, i))
  for split in data_set_splits:
    data_sets[split[0]] = []

  for i in range(1, len(chart_lines)):
    line = chart_lines[i].replace('"','')
    if line == '': continue

    els = line.split(',')
    for j in range(0, len(data_set_splits)):
      split = data_set_splits[j]
      if j+1 == len(data_set_splits):
        data_sets[split[0]].append(els[split[1]:])
      else:
        data_sets[split[0]].append(els[split[1]:data_set_splits[j+1][1]])

  dfs = {}
  for split in data_set_splits:
    # df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    del df['pen down?']
    chart_props['color'][split[0]] = df['color'].iloc[1]
    del df['color']
    # del df['x']
    dfs[split[0]] = df

  return (model_props, chart_props, dfs)

'''
Read multiple NetLogo chart export files and plot all of them on a single
Matplotlib plot.

:param in_path: The directory to search for files in.
:param in_filename: A piece of filename that indicates which files to parse
in the process. This should usually be the name of the chart in the NetLogo file.
'''
def process_multi_chart_data(in_path, in_filename='percent-agent-beliefs'):
  props = []
  multi_data = []
  run_ids = []
  print(f'process_multi_chart_data for {in_path}/{in_filename}')
  for file in os.listdir(in_path):
    if in_filename in file:
      run_id = file[0: file.index('_')]
      data = process_chart_data(f'{in_path}/{file}')
      model_params = data[0]
      props.append(data[1])
      multi_data.append(data[2])
      run_ids.append(run_id)

  means = { key: [] for key in multi_data[0].keys() }
  vector_length = int(model_params['tick-end'])
  for data in multi_data:
    for key in data.keys():
      data_vector = np.array(data[key]['y']).astype('float32')
      
      # Ensuring length of vector is equal to num ticks so the
      # vectors can stack on each other
      if len(data_vector) < vector_length:
        data_vector = np.append(data_vector, [ data_vector[-1] for i in range(vector_length - len(data_vector)) ])
      elif len(data_vector) > vector_length:
        data_vector = data_vector[:vector_length]

      if means[key] == []:
        means[key] = data_vector
      else:
        means[key] = np.vstack([means[key], data_vector])

  final_props = props[0]
  props_y_max = np.array([ float(prop['y max']) for prop in props ])
  final_props['y max'] = props_y_max.max()
  return (means, final_props, model_params, run_ids)

'''
Given some multi-chart data, plot it and save the plot.

:param multi_data: Data with means and std deviations for each point.
:param props: Properties object for the plotting.
:param out_path: A path to save the results in.
:param out_filename: A filename to save results as, defaults to 'aggregate-chart'
:param show_plot: Whether or not to display the plot before saving.
'''
def plot_multi_chart_data(types, multi_data, props, out_path, out_filename='aggregate-chart', show_plot=False):
  if PLOT_TYPES.LINE in types:
    plot = plot_nlogo_multi_chart_line(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_line.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.STACK in types:
    plot = plot_nlogo_multi_chart_stacked(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_stacked.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.HISTOGRAM in types:
    plot = plot_nlogo_multi_chart_histogram(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_histogram.png')
    if show_plot: plt.show()
    plt.close()

'''
Plot multiple NetLogo chart data sets on a single plot. 

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A dictionary (keyed by line) of matrices where each row is one simulation's worth of data points.
'''
def plot_nlogo_multi_chart_stacked(props, multi_data):
  init_dist_width = 10

  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("Portion of agents who believe b")
  ax.set_xlabel("Time Step")


  multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))

  # To use Netlogo colors
  # line_color = lambda key: f"#{rgb_to_hex(NLOGO_COLORS[int(round(float(props['color'][key])))])}"

  # To use higher resolution colors
  resolution = int(max(multi_data_keys_int))+1
  line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/(resolution-1))*int(key)), 0, round((255/(resolution-1)) * int(key)) ])}"
  
  mean_vecs = []
  var_vecs = []
  rev_keys_int = sorted(multi_data_keys_int, reverse=True)
  rev_keys = list(map(lambda el: f'{el}', rev_keys_int))
  for key in rev_keys:
    mean_vec = multi_data[key].mean(0)
    var_vec = multi_data[key].var(0)

    # Add padding for the initial values so those are displayed in the graph
    mean_vec = np.insert(mean_vec, 0, [ mean_vec[0] for i in range(init_dist_width) ])

    mean_vecs.append(mean_vec)
    var_vecs.append(var_vec)
  
  ax.set_xlim([x_min-init_dist_width,len(mean_vecs[0])-init_dist_width])
  plt.stackplot(range(x_min-init_dist_width, len(mean_vecs[0])-init_dist_width), mean_vecs, colors=[ f'{line_color(c)}' for c in rev_keys ], labels=[ f'b = {b}' for b in rev_keys ])

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_multi_chart_line(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  ax.set_ylim([0, y_max])
  plt.yticks(np.arange(y_min, y_max, step=1))
  # plt.yticks(np.arange(y_min, y_max*1.1, step=y_max/10))
  plt.xticks(np.arange(x_min, x_max*1.1, step=5))
  ax.set_ylabel("% of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0)
    var_vec = multi_data[key].var(0)
    # print(var_vec)
    ax.plot(mean_vec, c=line_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{line_color(key)}44')
  
  return multi_data

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_histogram(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("# of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    bar_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0)
    var_vec = multi_data[key].var(0)
    # print(var_vec)
    ax.plot(mean_vec, c=bar_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{bar_color(key)}44')
  
  return multi_data
      
'''
From a NetLogo world export file, read in the simulation data for citizens
and media entities. They are stored in Pandas dataframes for further processing.

:param path: The path to the file to read data in from.
'''
def process_sim_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  lines = raw.split('\n')
  turtle_data = []
  for i in range(0, len(lines)):
    line = lines[i]
    if line == '"TURTLES"':
      while line.strip() != '':
        i += 1
        line = lines[i]
        turtle_data.append(line.replace('""','"').split(','))

  turtle_data[0] = list(map(lambda el: el.replace('"',''), turtle_data[0]))
  turtle_df = pd.DataFrame(data=turtle_data[1:], columns=turtle_data[0])

  unneeded_cols = ['color', 'heading', 'xcor', 'ycor', 'label', 'label-color', 'shape', 'pen-size', 'pen-mode', 'size','hidden?']
  citizen_delete = ['media-attrs','messages-sent']
  media_delete = ['messages-heard','brain','messages-believed']

  for col in unneeded_cols:
    del turtle_df[col]

  citizen_df = turtle_df[turtle_df['breed'] == '"{breed citizens}"']
  media_df = turtle_df[turtle_df['breed'] == '"{breed medias}"']

  for col in citizen_delete:
    del citizen_df[col]
  for col in media_delete:
    del media_df[col]

  return (citizen_df, media_df)

'''
Get a relevant set of statistics about the citizens' ending state
after the simulation was run: which messages they heard

:param citizen_df: A dataframe containing citizen data.
'''
def citizen_message_statistics(citizen_df, media_df):
  messages = {}
  # Generate a data frame for media messages
  for m in media_df.iterrows():
    m_sent = nlogo_mixed_list_to_dict(m[1]['messages-sent'])
    messages.update(m_sent)
  for m_id, val in messages.items():
    val['id'] = int(m_id.strip())

  message_vals = list(messages.values())
  messages_df = pd.DataFrame(data=message_vals, columns=list(message_vals[0].keys()))

  # Generate citizen data frames relevant for statistics
  heard_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-heard'])
    flat_heard = []
    for timestep,message_ids in parsed.items():
      flat_heard.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
    df = pd.DataFrame(flat_heard)
    heard_dfs[int(citizen[1]['who'].replace('"',''))] = df
  
  believed_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-believed'])
    flat_believed = []
    if not type(parsed) == list:
      for timestep,message_ids in parsed.items():
        flat_believed.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
      df = pd.DataFrame(flat_believed)
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = df
    else:
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = pd.DataFrame()
  
  # Analyze the data frames for some statistical measures (per citizen)
  # - Total heard
  # - Total believed
  # - Ratio of believed/heard
  # - Totals heard broken down by partisanship & ideology
  # - Totals believed broken down by partisanship & ideology
  # - Totals heard broken down by virus belief
  # - Totals believed broken down by virus belief
  # - Somehow get at beliefs over time?

  per_cit_stats = {}
  for row in citizen_df.iterrows():
    citizen = row[1]
    cit_id = int(citizen['who'].replace('"',''))
    per_cit_stats[cit_id] = per_citizen_stats(cit_id, messages_df, heard_dfs[cit_id], believed_dfs[cit_id])

  # Analyze some group-level measures
  # - Aggregate by citizen's partisan/ideology pair
  # - 

  aggregate_stats = citizens_stats(citizen_df, per_cit_stats)

  return (messages_df, heard_dfs, believed_dfs, per_cit_stats, aggregate_stats)

'''
Generate some statistical measures based on the aggregate view of the citizenry.

:param cit_df: A dataframe containing data for each citizen.
:param per_cit_stats: A dictionary of statistical measures calculated
for each citizen. This is generated from the `citizen_stats()` function.
'''
def citizens_stats(cit_df, per_cit_stats):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  virus_believe_vals = [ -1, 0, 1 ]

  pi_keyed_dict = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  virus_belief_keyed_dict = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }

  total_by_p = { p: 0 for p in partisanships }
  total_by_i = { i: 0 for i in ideologies }
  total_by_p_i = pi_keyed_dict.copy()

  heard_by_cit_p_i = { (prod[0], prod[1]): [] for prod in itertools.product(partisanships, ideologies) }
  believed_by_cit_p_i = deepcopy(heard_by_cit_p_i)
  # This will be a dict of { (citizen_p, citizen_i): { (message_p, message_i):
  # [ list of (p,i) message heard ] } } - i.e. a dictionary of message types
  # heard by political group
  heard_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }
  believed_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }

  virus_bel_counts = virus_belief_keyed_dict.copy()
  virus_bel_totals = { (prod[0], prod[1], prod[2]): [] for prod in itertools.product(virus_believe_vals, repeat=3) }

  ending_beliefs_by_p_i = { (prod[0], prod[1]): virus_belief_keyed_dict.copy() for prod in itertools.product(partisanships, ideologies) }
  # Similarly to above, this will be a dict of { (cit_p, cit_i): {
  # (virus_beliefs...): [ list of per-citizen (vg,vs,vd) messages heard ] } }
  heard_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }
  believed_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }

  for cit in cit_df.iterrows():
    citizen = cit[1]
    cit_id = int(citizen['who'].replace('"',''))
    brain = nlogo_mixed_list_to_dict(citizen['brain'])
    stats = per_cit_stats[cit_id]

    pi_tup = pi_tuple(brain)
    virus_tup = virus_tuple(brain)

    total_by_i[int(brain['I'])] += 1
    total_by_p[int(brain['P'])] += 1
    total_by_p_i[pi_tup] += 1
    heard_by_cit_p_i[pi_tup].append(stats['total_heard'])
    believed_by_cit_p_i[pi_tup].append(stats['total_believed'])
    for message_pi_tup in stats['heard_by_p_i'].keys():
      heard_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['heard_by_p_i'][message_pi_tup])
      believed_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['believed_by_p_i'][message_pi_tup])

    virus_bel_counts[virus_tup] += 1
    ending_beliefs_by_p_i[pi_tup][virus_tup] += 1
    for message_virus_tup in stats['heard_by_virus_bel'].keys():
      heard_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['heard_by_virus_bel'][message_virus_tup])
      believed_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['believed_by_virus_bel'][message_virus_tup])
  
  heard_sum_by_p_i = { pi: summary_statistics(heard_by_cit_p_i[pi]) for pi in heard_by_cit_p_i.keys() }
  believed_sum_by_p_i = { pi: summary_statistics(believed_by_cit_p_i[pi]) for pi in believed_by_cit_p_i.keys() }

  heard_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_pi_given_cit_pi.keys():
    entry = heard_by_pi_given_cit_pi[pi]
    heard_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  believed_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_pi_given_cit_pi.keys():
    entry = believed_by_pi_given_cit_pi[pi]
    believed_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  heard_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_virus_bel_given_pi.keys():
    entry = heard_by_virus_bel_given_pi[pi]
    heard_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }

  believed_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_virus_bel_given_pi.keys():
    entry = believed_by_virus_bel_given_pi[pi]
    believed_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }
  
  stats_given_pi = pi_keyed_dict.copy()
  for pi in stats_given_pi.keys():
    stats_given_pi[pi] = {}
    stats_given_pi[pi]['n'] = total_by_p_i[pi]
    stats_given_pi[pi]['total_heard'] = heard_sum_by_p_i[pi]
    stats_given_pi[pi]['total_believed'] = believed_sum_by_p_i[pi]
    stats_given_pi[pi]['ending_beliefs'] = ending_beliefs_by_p_i[pi]
    stats_given_pi[pi]['heard_stats_by_pi'] = heard_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_pi'] = believed_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['heard_stats_by_virus'] = heard_sum_by_virus_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_virus'] = believed_sum_by_virus_given_pi[pi]

  return stats_given_pi

'''
Generate some statistics for each citizen in the simulation report data.
Measures that are reported:
- Total messages heard and believed
- Believed/heard ratio
- Messages heard & believed by (partisan,ideology) pair
- Messages heard & believed by virus-belief combination
'''
def per_citizen_stats(cit_id, messages_df, heard_df, believed_df):
  cit_stats = {}
  cit_stats['total_heard'] = len(heard_df)
  cit_stats['total_believed'] = len(believed_df)
  cit_stats['bel_heard_ratio'] = cit_stats['total_believed']/cit_stats['total_heard']

  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  heard_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }

  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_p_i[pi_tuple(message)] += 1

  # (P, I) tuples
  believed_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_p_i[pi_tuple(message)] += 1
  cit_stats['heard_by_p_i'] = heard_by_p_i
  cit_stats['believed_by_p_i'] = believed_by_p_i

  # (VG, VG, VD) tuples
  virus_believe_vals = [ -1, 0, 1 ]
  heard_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_virus_bel[virus_tuple(message)] += 1
  believed_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_virus_bel[virus_tuple(message)] += 1
  cit_stats['heard_by_virus_bel'] = heard_by_virus_bel
  cit_stats['believed_by_virus_bel'] = believed_by_virus_bel

  return cit_stats

def group_stats_by_attr(group_stats, attr):
  return { pi: val[attr] for pi,val in group_stats.items() }

'''
Return a tuple of partisanship and ideology attributes from a given object.

:param obj: Some object to fetch parameters from.
'''
def pi_tuple(obj): return (int(obj['P']),int(obj['I']))

'''
Return a tuple of virus-related beliefs from a given object.

:param obj: Some object to fetch parameters from.
'''
def virus_tuple(obj): return (int(obj['VG']),int(obj['VS']),int(obj['VD']))

def plot_stats_means(stats_data, title, path):
  plot_and_save_series({ key: val[0] for key,val in stats_data.items() }, title, path, 'bar')

def pi_data_charts(stats_data, attr, replace, title_w_replace, path_w_replace):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  pi_keys = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for key in pi_keys:
    plot_stats_means(stats_data[key][attr], title_w_replace.replace(replace, str(key)), path_w_replace.replace(replace, f'{key[0]}-{key[1]}'))

def corr_multi_data(multi_data_1, multi_data_2, method='pearson'):
  '''
  Calculate correlations between two sets of multi data.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.

  :return: Correlation values per belief value.
  '''
  m1_means = { key: multi_data_1[key].mean(0) for key in multi_data_1 }
  m2_means = { key: multi_data_2[key].mean(0) for key in multi_data_2 }

  rs = {}
  for key in multi_data_1:
    df = pd.DataFrame({ 'data1': m1_means[key], 'data2': m2_means[key] })
    # Uncomment if you need to investigate the df
    # rs[key] = {}
    # rs[key]['df'] = df
    # rs[key]['corr'] = df.corr(method=method).iloc[0,1]
    rs[key] = df.corr(method=method).iloc[0,1]
  return rs

def aggregate_corr(corr_by_bel):
  '''
  Generate an average correlation across correlations by belief value.

  :param corr_by_bel: A dictionary keyed by belief value of correlation values.
  '''
  non_nan = np.array(list(corr_by_bel.values()))
  non_nan = non_nan[np.logical_not(np.isnan(non_nan))]
  return non_nan.sum() / len(non_nan)

def chi_sq_test_multi_data(multi_data_1, multi_data_2, N):
  '''
  Perform a chi squared test on two sets of multi data for each timestep in the simulation data. 

  NOTE: This converts agent population percentages to total numbers and pads by 1
  in order to circumnavigate sampling 0 agents.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.
  :param N: The number of agents in the simulation.

  :returns: Returns the chi2 timeseries data.
  '''

  m1_means = [ multi_data_1[key].mean(0) for key in multi_data_1 ]
  m2_means = [ multi_data_2[key].mean(0) for key in multi_data_2 ]

  data = []
  for timestep in range(len(m1_means[0])):
    data.append([])
    # Append on lists of the values for each belief at timestep t 
    data[timestep].append([ m1_means[bel][timestep] for bel in range(len(m1_means)) ])
    data[timestep].append([ m2_means[bel][timestep] for bel in range(len(m2_means)) ])
  
  for data_t in data:
    for i in range(len(data_t[0])):
      data_t[0][i] = round(N * data_t[0][i] + 1)
      data_t[1][i] = round(N * data_t[1][i] + 1)
  
  chi2_data = [ chi2_contingency(data_t) for data_t in data ]
  # TODO: CHANGE THIS BACK
  # return chi2_data
  return (data, chi2_data)

def chi_sq_global(chi2_data):
  '''
  Convert a timeseries of chi squared test data into a global measure of how many
  entries in the time series are statistically independent. Higher values indicate
  higher levels of independence.

  :param chi2_data: An array of timeseries chi squared data from the scipy test.
  '''
  data = np.array([ el[1] for el in chi2_data ])
  return (data <= 0.05).sum() / len(data)


def plot_chi_sq_data(chi2_data, props, title, out_path, out_filename):
  '''
  Plot a time series calculation of chi squared measures per timestep.

  :param chi2_data: The timeseries data from running chi squared stats on belief data.
  :param props: The simulation properties to pull time data from.
  :param title: Text to title the plot with.
  '''
  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.0])
  y_min = 0
  y_max = 1.0
  x_min = int(props['x min'])
  x_max = int(props['x max'])
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.05))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("p value")
  ax.set_xlabel("Time Step")
  ax.set_title(f'{title}')
 
  ax.plot([ data[1] for data in chi2_data ])
  plt.savefig(f'{out_path}/{out_filename}')
  plt.close()

def analyze_high_spread_frequency(peak_spread_data, graph):
  '''
  Do an analysis to find out how often certain high-spreading agents
  spread belief to others across all the simulation runs.

  :param peak_spread_data: The result of calling `analyze_spread_peak_df`
  '''
  high_spread_frequency = { }
  for run_id,data in peak_spread_data.items():
    high_spreaders = data[1]
    for tick,spreader_set in high_spreaders.items():
      if not spreader_set: continue
      for spreader in spreader_set:
        spreader_data = spreader.split(',')
        agent_name = spreader_data[0]
        degree = spreader_data[1]
        sent_to = spreader_data[2]
        if agent_name not in high_spread_frequency:
          high_spread_frequency[agent_name] = { 'degree': degree, 'runs': { run_id: sent_to } }
        else:
          high_spread_frequency[agent_name]['runs'][run_id] = sent_to
  frequency_percents = {
    agent_name: { 
        'degree': data['degree'],
        'degree_percentile': percentileofscore(list(nx.degree(graph)), data['degree']),
        'percent_runs': len(data['runs']) / len(peak_spread_data)
      } for agent_name, data in high_spread_frequency.items()
  } 
  freq_percents_sorted = {agent_name: data for agent_name, data in sorted(frequency_percents.items(), key=lambda item: item[1]['percent_runs'])}
  return freq_percents_sorted

def analyze_dynamic_spread_peak_df(df, columns, sim_output_dir):
  analyzed_data = {}
  for row in df.iterrows():
    data = row[1]
    col_values = [ data[col] for col in columns ]
    col_string = '-'.join(col_values)

    # Read in the graph
    (cit, cit_social, media_arr, media_sub_arr)= read_graph(f'{sim_output_dir}/graphs/{col_string}.csv')
    graph = read_graph_to_nx_with_media(cit, cit_social, media_arr, media_sub_arr)
    # Read in the adoption data
    col_dir_string = '/'.join(col_values)

    adoption_data = None
    with open(f'{sim_output_dir}/{col_dir_string}/{data["run_id"]}_messages_adopted.json','r') as f:
      adoption_data = json.load(f)

    link_formation_data = None
    with open(f'{sim_output_dir}/{col_dir_string}/{data["run_id"]}_links_formed.json','r') as f:
      adoption_data = json.load(f)
    
    print(f'analyzing run {data["run_id"]}')
    analyzed_data[data['run_id']] = analyze_dynamic_spread_peak(data['data'], adoption_data, link_formation_data, graph)

  return analyzed_data

def analyze_static_spread_peak_df(df, columns, sim_output_dir):
  analyzed_data = {}
  for row in df.iterrows():
    data = row[1]
    col_values = [ data[col] for col in columns ]
    col_string = '-'.join(col_values)

    # Read in the graph
    (cit, cit_social, media_arr, media_sub_arr)= read_graph(f'{sim_output_dir}/graphs/{col_string}.csv')
    graph = read_graph_to_nx_with_media(cit, cit_social, media_arr, media_sub_arr)
    # Read in the adoption data
    col_dir_string = '/'.join(col_values)

    adoption_data = None
    with open(f'{sim_output_dir}/{col_dir_string}/{data["run_id"]}_messages_adopted.json','r') as f:
      adoption_data = json.load(f)
    
    print(f'analyzing run {data["run_id"]}')
    analyzed_data[data['run_id']] = analyze_static_spread_peak(data['data'], adoption_data, graph)

  return analyzed_data

def agent_id_from_name(agent_name):
  return agent_name.replace('(citizen ','').replace(')','') if 'citizen' in agent_name else agent_name.replace('(media ','').replace(')','')

def organizing_graph_at_tick(graph, link_formation_data, tick):
  for t,links_formed in sorted(link_formation_data).items():
    if t <= tick:
      for organizer,organized in links_formed.items():
        graph.add_edge(agent_id_from_name(organizer), agent_id_from_name(organized))
        graph.add_edge(agent_id_from_name(organized), agent_id_from_name(organizer))
  return graph

def analyze_dynamic_spread_peak(spread_data, adoption_data, link_formation_data, graph):
  '''
  :param adoption_data: JSON data about adoption in format { tick: [ { adopter1: sender1, adopter2: sender2, ... } ] }
  '''
  degree_from_agent_name = lambda agent_name, tick: nx.degree(organizing_graph_at_tick(graph,link_formation_data,tick))[int(agent_id_from_name(agent_name))] if int(agent_id_from_name(agent_name)) in dict(nx.degree(graph)) else -1

  # # Sanity check
  # for tick, adopters in adoption_data.items():
  #   for adopter, sender in adopters.items():
  #     adopter_id = int(agent_id_from_name(adopter))
  #     sender_id = int(agent_id_from_name(sender))
  #     if adopter_id not in dict(nx.degree(graph)):
  #       print(f'{adopter} in adopters but not in graph')
  #     elif sender_id not in dict(nx.degree(graph)):
  #       print(f'{sender} in senders but not in graph')

  adoption_data = {
    # tick: { dict of adopters & senders }
    int(tick): {
      # adopter_name,adopter_degree: sender_name,sender_degree
      f'{adopter},deg {degree_from_agent_name(adopter,tick)}': f'{sender},deg {degree_from_agent_name(sender,tick)}' for adopter, sender in adopters.items()
    } for tick,adopters in adoption_data.items()
  }
  
  peak_tick = list(spread_data).index(max(spread_data))

  around_peak_threshold = 3
  adopters_around_peak = { tick: adopters for tick,adopters in adoption_data.items() if (tick >= peak_tick - around_peak_threshold and tick <= peak_tick + around_peak_threshold) }
  
  high_spread_threshold = 10
  high_spread_agents = {
    tick: set([
      f'{sender},sent to {(np.array(list(adopters.values())) == sender).sum()}' for adopter, sender in adopters.items() if (np.array(list(adopters.values())) == sender).sum() >= high_spread_threshold
     ]) for tick, adopters in adopters_around_peak.items()
  }

  return (adopters_around_peak, high_spread_agents)

def analyze_static_spread_peak(spread_data, adoption_data, graph):
  '''
  :param adoption_data: JSON data about adoption in format { tick: [ { adopter1: sender1, adopter2: sender2, ... } ] }
  '''
  agent_id_from_name = lambda agent_name: agent_name.replace('(citizen ','').replace(')','') if 'citizen' in agent_name else agent_name.replace('(media ','').replace(')','')
  degree_from_agent_name = lambda agent_name: nx.degree(graph)[int(agent_id_from_name(agent_name))] if int(agent_id_from_name(agent_name)) in dict(nx.degree(graph)) else -1

  # # Sanity check
  # for tick, adopters in adoption_data.items():
  #   for adopter, sender in adopters.items():
  #     adopter_id = int(agent_id_from_name(adopter))
  #     sender_id = int(agent_id_from_name(sender))
  #     if adopter_id not in dict(nx.degree(graph)):
  #       print(f'{adopter} in adopters but not in graph')
  #     elif sender_id not in dict(nx.degree(graph)):
  #       print(f'{sender} in senders but not in graph')

  adoption_data = {
    # tick: { dict of adopters & senders }
    int(tick): {
      # adopter_name,adopter_degree: sender_name,sender_degree
      f'{adopter},deg {degree_from_agent_name(adopter)}': f'{sender},deg {degree_from_agent_name(sender)}' for adopter, sender in adopters.items()
    } for tick,adopters in adoption_data.items()
  }
  
  peak_tick = list(spread_data).index(max(spread_data))

  around_peak_threshold = 3
  adopters_around_peak = { tick: adopters for tick,adopters in adoption_data.items() if (tick >= peak_tick - around_peak_threshold and tick <= peak_tick + around_peak_threshold) }
  
  high_spread_threshold = 10
  high_spread_agents = {
    tick: set([
      f'{sender},sent to {(np.array(list(adopters.values())) == sender).sum()}' for adopter, sender in adopters.items() if (np.array(list(adopters.values())) == sender).sum() >= high_spread_threshold
     ]) for tick, adopters in adopters_around_peak.items()
  }

  return (adopters_around_peak, high_spread_agents)

"""
##################
EXPERIMENT-SPECIFIC
ANALYSIS
##################
"""

class PLOT_TYPES(Enum):
  LINE = 0
  STACK = 1
  HISTOGRAM = 2

def process_exp_outputs(param_combos, plots, path):
  '''
  Process the output of a NetLogo experiment, aggregating all results
  over simulation runs and generating plots for them according to
  all the parameter combinations denoted in param_combos.
  
  :param param_combos: A list of parameters where their values are
  lists (e.g. [ ['simple','complex'], ['default', 'gradual'] ])
  :param plots: A list of dictionaries keyed by the name of the NetLogo
  plot to process, with value of a list of PLOT_TYPE
  (e.g. { 'polarization': [PLOT_TYPES.LINE], 'agent-beliefs': [...] })
  :param path: The root path to begin processing in.
  '''
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  if not os.path.isdir(f'{path}/results'):
    os.mkdir(f'{path}/results')

  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      plot_multi_chart_data(plot_types, multi_data, props, f'{path}/results', f'{"-".join(combo)}_{plot_name}-agg-chart')

def get_all_multidata(param_combos, plots, path):
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  if not os.path.isdir(f'{path}/results'):
    os.mkdir(f'{path}/results')

  multi_datas = {}
  multi_data_ids = {}
  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params, run_ids) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      multi_datas[(combo,plot_name)] = multi_data
      multi_data_ids[(combo,plot_name)] = run_ids
  return (multi_datas, props, model_params, multi_data_ids)

def process_belief_spread_exp_test_results(path):
  simple_spread_chance = [ '0.5' ]
  ba_m = ['3']
  cit_media_influence = [ '0.01']
  cit_cit_influence = [ '0.01']
  repetition = ['0','1','2','3']

  process_exp_outputs(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)

def process_belief_spread_exp_results(path):
  simple_spread_chance = [ '0.01', '0.05']
  # simple_spread_chance = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  ba_m = ['3','10']
  cit_media_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  cit_cit_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  repetition = ['0','1','2','3','4']

  process_exp_outputs(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)

def base_model_sweep_results_to_df(path):
  simple_spread_chance = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  ba_m = ['1','2','3','5','10','20','50']
  cit_cit_influence = ['1']
  cit_media_influence = ['1']
  repetition = ['0','1','2','3','4']

  measures = ['new-beliefs']
  df_columns = { "new-beliefs": ['n','spread-type','simple-spread-chance','graph-type','ba-m','citizen-citizen-influence', 'citizen-media-influence','repetition'] }
  multidata_key_params = ['simple-spread-chance','ba-m','citizen-citizen-influence', 'citizen-media-influence', 'repetition']

  (multidata, props, params, multidata_ids) = get_all_multidata(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)
  return multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids)

def static_influence_monte_carlo_1_results_to_df(path):
  simple_spread_chance = [ '0.75' ]
  ba_m = ['10']
  cit_cit_influence = [ '0.75' ]
  cit_media_influence = [ '0.01']
  repetition = ['0']

  measures = ['new-beliefs']
  df_columns = { "new-beliefs": ['n','spread-type','simple-spread-chance','graph-type','ba-m','cit-media-influence','cit-cit-influence','repetition'] }
  multidata_key_params = ['simple-spread-chance','ba-m','cit-media-influence','cit-cit-influence','repetition']

  (multidata, props, params, multidata_ids) = get_all_multidata(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)
  return multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids)

def static_influence_sweep_results_to_df(path):
  simple_spread_chance = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  ba_m = ['3','10']
  cit_cit_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  cit_media_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  repetition = ['0','1','2','3','4']

  measures = ['new-beliefs']
  df_columns = { "new-beliefs": ['n','spread-type','simple-spread-chance','graph-type','ba-m','cit-media-influence','cit-cit-influence','repetition'] }
  multidata_key_params = ['simple-spread-chance','ba-m','cit-media-influence','cit-cit-influence','repetition']

  (multidata, props, params, multidata_ids) = get_all_multidata(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)
  return multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids)

def static_no_organizing_media_connect_sweep_results_to_df(path):
  simple_spread_chance = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  ba_m = ['3','10']
  cit_cit_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  cit_media_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  repetition = ['0','1','2','3','4']

  measures = ['new-beliefs']
  df_columns = { "new-beliefs": ['n','spread-type','simple-spread-chance','graph-type','ba-m','cit-media-influence','cit-cit-influence','repetition'] }
  multidata_key_params = ['simple-spread-chance','ba-m','cit-media-influence','cit-cit-influence','repetition']

  (multidata, props, params, multidata_ids) = get_all_multidata(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)
  return multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids)

def static_organizing_media_connect_sweep_results_to_df(path):
  simple_spread_chance = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  ba_m = ['3','10']
  cit_cit_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  cit_media_influence = [ '0.01', '0.05', '0.1', '0.25', '0.5', '0.75' ]
  organizing_capacity = ['1','5']
  organizing_strategy = ['neighbors-of-neighbors','high-degree-media','high-degree-citizens','high-degree-cit-and-media']
  repetition = ['0','1','2','3','4']

  measures = ['new-beliefs']
  df_columns = { "new-beliefs": ['n','spread-type','simple-spread-chance','graph-type','ba-m','cit-media-influence','cit-cit-influence','organizing-capacity','flint-organizing-capacity','repetition'] }
  multidata_key_params = ['simple-spread-chance','ba-m','cit-media-influence','cit-cit-influence','organizing-capacity','flint-organizing-strategy','repetition']

  (multidata, props, params, multidata_ids) = get_all_multidata(
    [simple_spread_chance,ba_m,cit_media_influence,cit_cit_influence,organizing_capacity,organizing_strategy,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'new-beliefs': [PLOT_TYPES.LINE]},
    path)
  return multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids)

def multidata_to_dataframes(measures, df_columns, multidata, multidata_key_params, props, params, multidata_ids):
  '''
  Convert experiment multidata into a usable dataframe format. This
  returns a dictionary of dataframes keyed by measure name (the name
  of the graph used to output results in NetLogo).

  :param measures: A list of NetLogo graph outputs to collect data for.
  Each of these will yield a separate dataframe.
  :param df_columns: A dictionary of column title arrays keyed by
  the measure they should be for (e.g. {'new-beliefs': ['N','ba-m',...]})
  :param multidata: A collection of multidata for a given experiment.
  :param multidata_key_params: A list (in order) of parameters that were
  varied during the experiment, and are used as keys for the multidata.
  :param props:
  :param params: Simulation parameters for the experiment.
  :param multidata_ids: Random ids generated for each multidata run
  per param combo and graph name.
  '''

  dfs = { measure: pd.DataFrame(columns=df_columns[measure]+['data','run_id']) for measure in measures }

  for measure in measures:
    data = { key: value for (key,value) in multidata.items() if key[1] == measure }
    df = dfs[measure]

    for (param_measure_combo, data_points) in data.items():
      param_combo = param_measure_combo[0]
      # TODO: Maybe this should be generalized somehow to incorporate
      # different pen names?
      data_points = data_points['default']
      for i in range(len(data_points)):
        data_point = data_points[i]
        data_row = [ param_combo[multidata_key_params.index(col)] if col in multidata_key_params else params[col] for col in df_columns[measure] ]
        df.loc[len(df.index)] = data_row + [data_point,multidata_ids[param_measure_combo][i]]
  
  return dfs

def add_graph_data_to_dataframe(df, columns, path):
  '''
  Add pieces of graph data to an existing experiment dataframe. This
  loops through all parameter combinations for given columns
  based on the combinations present in the dataframe.

  :param df: An experiment result data frame to add data to.
  :param columns: The parameters to use for filename generation to read
  in graph data.
  :param path: The directory to look for graph data in.
  '''
  new_df = df.copy()
  new_cols = {
    'num_media': []
  }
  for row in df.iterrows():
    data = row[1]
    col_values = [ data[col] for col in columns ]
    col_string = '-'.join(col_values)
    (cit, cit_social, media_arr, media_sub_arr) = read_graph(f'{path}/{col_string}.csv')
    new_cols['num_media'].append(len(media_arr))
  for key, val in new_cols.items():
    new_df[key] = val
  return new_df