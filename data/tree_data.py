import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re





'''
NETLOGO PARSING
'''

def nlogo_list_to_arr(list_str):
    return [ el.replace('[', '').strip().split(' ') for el in list_str[1:len(list_str)-1].split(']') ]

#def nlogo_replace_agents(string, types):
#    for type in types:
#        string = string.replace(f'({type} ', f'{type}_')
#    return string.replace(')','')
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
  print(d)
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






def createdataframe(dataset):
    df =  pd.read_csv(dataset, sep='|' , engine='python')
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence', 'citizen-citizen-influence', 'flint-community-size', 'data']
    df.set_index('run', inplace=True)
    return df

def convertdata(data):
    data.strip()
    ndata = nlogo_parse_chunk(data)
    n2data = [elem.replace('.', '') for elem in ndata]
    print(len(n2data))
    list=[]
    for x in n2data:
        list.append(x.replace("\r\n", ""))
    print(list)
    print(len(list))
    return list

def convert_to_int(data):
    for i in range(0, len(data)):
        data[i] = int(data[i])
    return data


def test_run_data(dataset):
    df = createdataframe(dataset)
    run1=df.iloc[120]
    data=run1['data']
    print(data)
    finallist = convertdata(data)
    int_data = convert_to_int(finallist)
    print(int_data)
    ylength= len(int_data)
    ylist=[]
    for i in range(1,ylength + 1):
        timestep = i
        ylist.append(timestep)
    print(ylength)
    print(ylist)
    print(type(finallist))
    #plt.plot(finalist, ylist)
    #plt.show()
    #need to convert string to list of integers


#below works, but I can't get a good print out
#nlogo_mixed_list_to_dict_rec('belief-spread-exp-results.csv')
#make this work currently is nothing


#my code
test_run_data('belief-spread-exp-results.csv')


#Nick's code
    #get csv to string??

def activate_nicks_code(csv):
    df = pd.read_csv(csv, sep='|', engine='python')
    df.columns = ['run', 'n', 'spread-type', 'simple-spread-chance', 'graph-type', 'ba-m', 'citizen-media-influence',
                  'citizen-citizen-influence', 'flint-community-size', 'data']
    df.set_index('run', inplace=True)
    chunk = df['data']
    array = nlogo_list_to_arr(chunk)
    nlogo_mixed_list_to_dict(array)

#activate_nicks_code('belief-spread-exp-results.csv')


#need to delete '[' etc- should take another peak at Nick's code to see if i can get this to work- his parsing should be super helpful we just need to understand!!