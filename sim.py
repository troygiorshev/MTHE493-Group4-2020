'''
Simulation for MTHE 493 Group 4

Let's be 0 based for all of the indexing.

Using networkx for the graph
'''

import random

import matplotlib as mpl
from matplotlib import animation as ani
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
PROPORTION_S_THOUGHTS = 0.16
S_THOUGHTS_THRESHOLD = 0.7
S_THRESHOLD = 0.9

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

def adjust_for_real_stats():
    return NUM_NODES*PROPORTION_S_THOUGHTS

def init_nodes():
    num_unhealthy = int(adjust_for_real_stats())
    for node in nodes[0:num_unhealthy - 1]:
        node[0] = int(10 * S_THOUGHTS_THRESHOLD)
        node[1] = int(10 - 10 * S_THOUGHTS_THRESHOLD)
    '''Initialize each node with 10 balls, with between 0 and 5 red balls'''
    for node in nodes[num_unhealthy:]:
        tmp = random.randint(0, 5)  # 0 to 5 inclusive
        node[0] = tmp
        node[1] = 10 - tmp


def set_delta(nodes,neighbors):
    '''Set delta for each time'''
    sum_R = 0
    for node in range(NUM_NODES):
        sum_R += nodes[node][0]
    return int(sum_R/neighbors)


def calculate_proportions(print_out=False):
    global nodes
    prop = []
    for node in nodes:
        tmp = 100*node[0]/(node[0] + node[1])
        prop.append(tmp)
        if (print_out):
            print(f"{tmp:2.0f}%")
    return prop


def show_network(G, pos, prop):
    # Fix the colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    # Draw
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size = 50, vmin=0, vmax=100, node_color = prop, cmap=plt.cm.Reds, edgecolors='black')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Reds))
    cbar.set_label("Proportion of Red")


def updateFunc(step, G, pos):
    global nodes
    print()
    plt.clf()   # Without this, the colorbars act all weird
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
    # Print
    print(nodes)
    prop_current = calculate_proportions()
    show_network(G, pos, prop_current)
    plt.title(f"Step {step+1}")
    #plt.show()


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
    prop_before = calculate_proportions(print_out=True)

    # Show network
    print()
    print(nx.info(G))
    print(f"Density: {nx.density(G)}")
    print(f"Diameter: {nx.diameter(G)}")

    # Current Status
    show_network(G, pos, prop_before)
    plt.title("Before")
    plt.show()

    # Loop
    fig = plt.figure()
    animator = ani.FuncAnimation(fig, updateFunc, fargs=(G,pos), interval=1000, frames=10, repeat=False)
    plt.show()


    # Print Result
    print()
    print("Proportion of Red Before")
    for val in prop_before:
        print(f"{val:2.0f}%")

    print()

    print("Proportion of Red After")
    prop_after = calculate_proportions(print_out=True)

    print("Got here")

    # Show network
    show_network(G, pos, prop_after)
    plt.title("After")
    plt.show()


if __name__ == "__main__":
    main()
