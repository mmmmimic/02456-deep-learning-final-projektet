'''
author: lmx
date: 11/17/2020
'''

# importation
import networkx as nx
import numpy
import sys
from glob import glob
from load_data import read_graphfile
from matplotlib import pyplot as plt
import torch
import math
from collections import Counter

#----------CONSTANTS----------
DATA_DIR = './data/' # parent folder 
DATA_NAME = 'DD'
############
NEW_DIR = DATA_DIR # new path
NAME = 'DDD' # new name
NUM = 10 # new graph number
############


# fetch graphs
# g = read_graphfile(DATA_DIR, DATA_NAME)
# assert NUM <= len(g),'exceed maximum graph number'
# nodes = list(map(lambda x: len(x.nodes), g))
# edges = list(map(lambda x: len(x.edges), g))

#-> PART 1
# fetch basic information 
base = '{}{}/{}_graph_indicator.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)

with open(old_dir, 'r') as f:
    c = f.read()
a = c.split('\n')
cnter = Counter(a)
cnter = list(zip(cnter.values(), cnter.keys()))
cnter = cnter[:-1]

def foo(x):
    return eval(x[1])
cnter = list(sorted(cnter, key=foo))
cnter = list(map(lambda x: x[0], cnter))

base = '{}{}/{}_A.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)

with open(old_dir, 'r') as f:
    c = f.read()
old_adj = c.split('\n')

def foo(x):
    x = x.split(',')
    try:
        assert 2 == len(x)
        return [eval(x[0]), eval(x[1])]
    except AssertionError:
        pass
    
old_adj = list(map(foo, old_adj))
old_adj = old_adj[:-1]

NODE_NUM = sum(list(cnter[:NUM]))

i = 0 
for i in range(len(old_adj)):
    tmp = old_adj[i]
    e0, e1 = tmp[0], tmp[1]
    if max([e0, e1]) > NODE_NUM:
        break

EDGE_NUM = i

print('There are {} graphs, {} nodes and {} edges.'.format(NUM, NODE_NUM, EDGE_NUM))

#-> PART 2
# Adjacent matrix
base = '{}{}/{}_A.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)
with open(old_dir, 'r') as f:
    c = f.read()
old_adj = c.split('\n')
new_adj = old_adj[:EDGE_NUM]
new_adj = '\n'.join(new_adj)
with open(new_dir, 'w') as f:
    f.write(new_adj)

print('Write Adjacent Matrix in {}'.format(new_dir))


#-> PART 3
# Graph indicator

base = '{}{}/{}_graph_indicator.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)

with open(old_dir, 'r') as f:
    c = f.read()
old_ind = c.split('\n')
new_ind = old_ind[:NODE_NUM]
new_ind = '\n'.join(new_ind)
with open(new_dir, 'w') as f:
    f.write(new_ind)

print('Write Graph Indicator in {}'.format(new_dir))


#-> PART 4
# Graph labels

base = '{}{}/{}_graph_labels.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)

with open(old_dir, 'r') as f:
    c = f.read()
old_label = c.split('\n')
new_label = old_label[:NUM]
new_label = '\n'.join(new_label)
with open(new_dir, 'w') as f:
    f.write(new_label)   

print('Write Graph Labels in {}'.format(new_dir))

#-> PART 5
# Node labels
base = '{}{}/{}_node_labels.txt'
old_dir = base.format(DATA_DIR, DATA_NAME, DATA_NAME)
new_dir = base.format(NEW_DIR, NAME, NAME)

with open(old_dir, 'r') as f:
    c = f.read()
old_label = c.split('\n')
new_label = old_label[:NODE_NUM]
new_label = '\n'.join(new_label)
with open(new_dir, 'w') as f:
    f.write(new_label)   

print('Write Node Labels in {}'.format(new_dir))