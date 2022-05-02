import numpy as np 

'''
Generic utilities file to keep track of useful functions.
'''

def dict_sort(d, reverse=False):
  return {key: value for key, value in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % tuple(rgb)

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
    