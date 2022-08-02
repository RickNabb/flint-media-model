from email import message
from enum import Enum
from random import *
from utils import *
from statistics import mean, variance, mode
from copy import deepcopy
from plotting import *
from nlogo_colors import *
import itertools
import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency, truncnorm
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt

def createdataframe(dataset):
    usersDf =  pd.read_csv('users.csv', sep='__'  , engine='python')