'''
Simulation for MTHE 493 Group 4

Let's be 0 based for all of the indexing.

Using networkx for the graph
'''

import random

import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from colorsys import hls_to_rgb

# Other imports maybe we'll use one day
import datetime as dt
import multiprocessing as mp
import pandas as pd
import seaborn as sns 

#random.seed(1234)   # Set random seed for reproducability

NUM_NODES = 23

# Values of each node.  [R,B]
nodes = np.array([[0,0]]*NUM_NODES)


def check_setup(G):
    '''Check that the variables make sense with each other.  Graph G'''
    assert(NUM_NODES == len(nodes))
    assert(NUM_NODES == len(G))


def connected_graph():
    '''V1 of making a connected graph'''
    # Adjacency Matrix
    adj = np.array([
        [0,1,1,1,1],
        [1,0,1,1,1],
        [1,1,0,1,1],
        [1,1,1,0,1],
        [1,1,1,1,0],
    ])
    return nx.from_numpy_array(adj)

def ba_graph():
    return nx.extended_barabasi_albert_graph(NUM_NODES, 2, 0, 0)


def init_nodes():
    '''Initialize each node with 10 balls, with between 0 and 5 red balls'''
    for node in nodes:
        tmp = random.randint(0,5)   # 0 to 5 inclusive
        node[0] = tmp
        node[1] = 10-tmp


def set_delta(nodes,neighbors):
    '''Set delta for each time'''
    sum_R = 0
    for node in range(NUM_NODES):
        sum_R += nodes[node][0]
    return int(sum_R/neighbors)


def main():
    global nodes
    '''Main setup and loop'''
    # Setup
    G = ba_graph()
    pos = nx.spring_layout(G)
    init_nodes()

    # Check
    check_setup(G)

    # Show values
    print(nodes)

    print()

    print("Proportion of Red Before")
    prop_before = []
    for node in nodes:
        tmp = 100*node[0]/(node[0] + node[1])
        prop_before.append(tmp)
        print(f"{tmp:2.0f}%")

    # Fix Colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    # Show network
    print()
    print(nx.info(G))
    print(f"Density: {nx.density(G)}")
    print(f"Diameter: {nx.diameter(G)}")
    nx.draw_networkx_edges(G, pos)
    test = nx.draw_networkx_nodes(G, pos, node_size = 50, vmin=0, vmax=100, node_color = prop_before, cmap=plt.cm.Reds, edgecolors='black')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Reds))
    cbar.set_label("Proportion of Red")
    plt.title("Before")
    plt.show()

    # Loop
    for j in range(10):
        print()
        new_nodes = nodes.copy() # Careful, copy the array!
        for i in range(NUM_NODES):
            # Make the super node
            super_node = [0,0]
            neighbors = 0
            for neighbor in G.neighbors(i):
                super_node[0] += nodes[neighbor][0]
                super_node[1] += nodes[neighbor][1]
                neighbors += 1
            # Draw from the super node
            rng = random.random()
            ball = 0 if rng < super_node[0] / (super_node[0] + super_node[1]) else 1
            delta = set_delta(new_nodes,neighbors)
            new_nodes[i][ball] += delta
        # I'm not sure this needs to be a copy, better safe than sorry
        nodes = new_nodes.copy()
        print(nodes)

    # Print Result
    print()
    print("Proportion of Red Before")
    for val in prop_before:
        print(f"{val:2.0f}%")

    print()

    print("Proportion of Red After")
    prop_after = []
    for node in nodes:
        tmp = 100*node[0]/(node[0] + node[1])
        prop_after.append(tmp)
        print(f"{tmp:2.0f}%")

    # Show network
    nx.draw_networkx_edges(G, pos)
    test = nx.draw_networkx_nodes(G, pos, node_size = 50, vmin=0, vmax=100, node_color = prop_after, cmap=plt.cm.Reds, edgecolors='black')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Reds))
    cbar.set_label("Proportion of Red")
    plt.title("After")
    plt.show()

if __name__ == "__main__":
    main()
