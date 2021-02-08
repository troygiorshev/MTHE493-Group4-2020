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

NUM_NODES = 150
PROPORTION_S_THOUGHTS = 0.16
S_THOUGHTS_THRESHOLD = 0.7 #change to based on sadness scale
S_THRESHOLD = 0.9

CLIQUE_SIZE = 5

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

def fill_houses():
    ROWS = 13
    COLS = 30
    # Houses by number of bedrooms
    layout = np.array([
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,4,3,3,3,3,3,3,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5,4,3,3,3,3,3,3,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,4,3,3,3,3,3,3,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,4,3,3,3,3,3,3,2,2,2,2],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,4,3,3,3,3,3,3,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1],
        [1,2,2,2,2,3,3,3,3,3,3,4,4,4,5,5,5,5,5,3,3,3,3,3,3,2,2,2,2,1]
        ])
    # Houses by number of people in them
    houses = np.full_like(layout, 0)

    people_left = NUM_NODES
    # First, fill every bedroom with a person
    for row, row_of_lots  in enumerate(layout):
        for col, lot in enumerate(row_of_lots):
            houses[row,col] = lot
            people_left -= lot
    
    # Randomly empty 5(ish) interior houses
    for _ in range(5):
        row = random.randint(1, ROWS-2)
        col = random.randint(1, COLS-2)
        people_left += houses[row,col]
        houses[row,col] = 0

    # Keep adding people randomly to the 4 and 5 person houses
    # While ensuring that no house goes over 15
    while people_left > 0:
        # This is a terribly slow way to do this
        # But it still happens pretty much instantly
        row = random.randint(0, ROWS-1)
        col = random.randint(0, COLS-1)
        if layout[row,col] < 4:
            continue
        if houses[row,col] >= 15:
            continue
        houses[row,col] += 1
        people_left -= 1
    
    #print(houses)

    # TODO surely there's a clearer way to initialize a 2x2 array...
    subgraphs = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for row, row_of_houses in enumerate(houses):
        for col, num_residents in enumerate(row_of_houses):
            if num_residents != 0:
                subgraphs[row][col] = nx.complete_graph(num_residents)
            else:
                subgraphs[row][col] = nx.null_graph()

    subgraphs_flat = []
    name_list = []
    for i, row_of_subgraphs in enumerate(subgraphs):
        for j, subgraph in enumerate(row_of_subgraphs):
            if len(subgraph) > 0:   # Dont add the empty houses
                subgraphs_flat.append(subgraph)
                name_list.append(f'h{i},{j}-')

    G = nx.union_all(subgraphs_flat, name_list)

    for i in range(ROWS):
        for j in range(COLS):
            if G.has_node(f'h{i},{j}-0'):
                if G.has_node(f'h{i+1},{j}-0'):
                    G.add_edge(f'h{i},{j}-0', f'h{i+1},{j}-0')
                if G.has_node(f'h{i},{j+1}-0'):
                    G.add_edge(f'h{i},{j}-0', f'h{i},{j+1}-0')

    #nx.draw_networkx(G)     # This takes a few seconds...
    #plt.show()

    return nx.convert_node_labels_to_integers(G)


# add up _____
#compute connectivity i.e. average degree (density of graph)

'''Not using this for now'''
def ba_graph():
    return nx.extended_barabasi_albert_graph(NUM_NODES, 2, 0, 0)

def clique_graph():
    '''Relaxed caveman graph returns a graph with `l` cliques of size `k`. Edges are
    then randomly rewired with probability `p` to link different cliques'''
    return nx.relaxed_caveman_graph(int(NUM_NODES/CLIQUE_SIZE), CLIQUE_SIZE, 0.5, seed=None)

def define_risk_levels():
    proportionRiskLevel = {}
    proportionRiskLevel[1] = 0.2
    proportionRiskLevel[2] = 0.2
    proportionRiskLevel[3] = NUM_NODES*PROPORTION_S_THOUGHTS
    proportionRiskLevel[4] = 0.2
    at_risk = sum(proportionRiskLevel.values())
    proportionRiskLevel[0] = 1 - at_risk
    return proportionRiskLevel

def init_nodes():
    riskLevels = define_risk_levels()
    num_unhealthy = int(riskLevels[4]*NUM_NODES)
    for node in nodes[0:num_unhealthy - 1]:
        node[0] = int(10 * S_THOUGHTS_THRESHOLD)
        node[1] = int(10 - 10 * S_THOUGHTS_THRESHOLD)
    '''Initialize rest of nodes with 10 balls, with between 0 and 5 red balls'''
    for node in nodes[num_unhealthy:]:
        tmp = random.randint(0, 5)  # 0 to 5 inclusive
        node[0] = tmp
        node[1] = 10 - tmp

def init_nodes_profiles():
    '''Initialize the nodes according to the following profiles.  Total 10 balls'''
    global nodes
    percentages = [20, 65, 15]
    proportions = [20, 45, 65]

    # Sanity Check
    assert(sum(percentages) == 100)

    numbers = [int(NUM_NODES*percentage/100) for percentage in percentages]

    part = 0
    for i, number in enumerate(numbers):
        for node in nodes[part : part + number]:
            node[0] = int(proportions[i] / 10)   # (x / 100) * 10 = x / 10
            node[1] = 10 - node[0]
        part += number
    
    # Now shuffle the nodes array
    np.random.shuffle(nodes)

def set_delta(neighbors):
    '''Set delta for each time.  Neighbors here should include the given node'''
    global nodes
    sum_R = 0
    for node in neighbors:
        sum_R += nodes[node][0]
    return int(sum_R/len(neighbors))


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
        # Remember "neighbors" for us includes the node
        neighbors = list(G[i])
        neighbors.append(i)
        super_node = [0,0]
        for neighbor in neighbors:
            super_node[0] += nodes[neighbor][0]
            super_node[1] += nodes[neighbor][1]
        # Draw from the super node
        rng = random.random()
        ball = 0 if rng < super_node[0] / (super_node[0] + super_node[1]) else 1
        delta = set_delta(neighbors)
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
    #G = clique_graph()
    pos = nx.spring_layout(G)
    init_nodes_profiles()

    print("Average node connectivity: ", nx.average_node_connectivity(G))

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

    # Show network
    show_network(G, pos, prop_after)
    plt.title("After")
    plt.show()

if __name__ == "__main__":
    main()
