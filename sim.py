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

# ============ Simulation Parameters and Global Variables ============

NUM_TIME_STEPS = 52
NUM_NODES = 150
PROPORTION_S_THOUGHTS = 0.16
S_THOUGHTS_THRESHOLD = 0.7 #change to based on sadness scale
S_THRESHOLD = 0.9

CLIQUE_SIZE = 5

# Values of each node's urn.  [R,B]
nodes = np.zeros((NUM_NODES,2), dtype=int)

metrics = []

# ================ Checks / Helper / Metric Functions ================

def check_setup(G):
    '''Check that the variables make sense with each other.  Graph G'''
    assert(NUM_NODES == len(nodes))
    assert(NUM_NODES == len(G))


def show_network(G, pos, prop):
    # Fix the colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    # Draw
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size = 50, vmin=0, vmax=100, node_color = prop, cmap=plt.cm.Reds, edgecolors='black')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Reds))
    cbar.set_label("Proportion of Red")

def iterative_mean(arr):
    '''We're starting to get overflow errors.
    This calculates the mean iteratively so that the sum doesn't get too large.
    '''
    avg = 0
    t = 1
    for x in arr:
        avg += (x - avg) / t
        t += 1
    return avg

def calculte_metrics(G):
    global nodes
    metrics = {}

    ## Network Susceptibility
    metrics["network susceptibility"] = iterative_mean(
        [node[0]/(node[0]+node[1]) for node in nodes]
    )

    # Calculate super nodes, we'll use these next
    super_nodes = []
    for i in range(len(nodes)):
        neighbors = list(G[i])
        neighbors.append(i)
        super_node = [0,0]
        for neighbor in neighbors:
            super_node[0] += nodes[neighbor][0]
            super_node[1] += nodes[neighbor][1]
        super_nodes.append(super_node)

    ## Network Exposure
    metrics["network exposure"] = iterative_mean(
        [super_node[0]/(super_node[0]+super_node[1]) for super_node in super_nodes]
    )

    ## Suicidal Thoughts
    metrics["suicial thoughts"] = len(
        [node for node in nodes if node[0]/(node[0]+node[1]) >= S_THOUGHTS_THRESHOLD]
    )

    return metrics

# ========================= Graph generators =========================

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


def houses_graph():
    assert(NUM_NODES == 1500)
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


def ba_graph():
    return nx.extended_barabasi_albert_graph(NUM_NODES, 2, 0, 0)


def clique_graph():
    '''Relaxed caveman graph returns a graph with `l` cliques of size `k`. Edges are
    then randomly rewired with probability `p` to link different cliques'''
    return nx.relaxed_caveman_graph(int(NUM_NODES/CLIQUE_SIZE), CLIQUE_SIZE, 0.5, seed=None)

def profiles():
    risk_factors = {}

# ========================= Node Initializers =========================

def define_risk_levels():
    proportionRiskLevel = {}
    proportionRiskLevel["med-low"] = 0.25
    proportionRiskLevel["med"] = 0.35
    proportionRiskLevel["med-high"] = 0.15 #NUM_NODES*PROPORTION_S_THOUGHTS
    proportionRiskLevel["high"] = 0.10
    #at_risk = sum(proportionRiskLevel.values)
    proportionRiskLevel["low"] = 0.15
    return proportionRiskLevel


def init_nodes():
    global nodes
    proportion_risk_levels = define_risk_levels()
    num_each_risk_level = {}
    for level in proportion_risk_levels: num_each_risk_level[level] = int(proportion_risk_levels[level]*NUM_NODES)
    risk_range = num_each_risk_level["low"]
    for node in nodes[0:risk_range-1]:
        tmp = random.randint(1, 2)
        node[0] = tmp
        node[1] = 10 - tmp
    risk_range += num_each_risk_level["med-low"]
    for node in nodes[risk_range - num_each_risk_level["med-low"]:risk_range-1]:
        tmp = random.randint(3, 4)
        node[0] = tmp
        node[1] = 10 - tmp
    risk_range += num_each_risk_level["med"]
    for node in nodes[risk_range - num_each_risk_level["med"]:risk_range - 1]:
        tmp = random.randint(5, 6)
        node[0] = tmp
        node[1] = 10 - tmp
    risk_range += num_each_risk_level["med-high"]
    for node in nodes[risk_range - num_each_risk_level["med-high"]:risk_range - 1]:
        tmp = random.randint(7, 8)
        node[0] = tmp
        node[1] = 10 - tmp
    risk_range += num_each_risk_level["high"]
    for node in nodes[risk_range - num_each_risk_level["high"]:risk_range - 1]:
        tmp = random.randint(8, 9)
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

# ========================= Delta Function =========================

def set_delta(neighbors):
    '''Set delta for each time.  Neighbors here should include the given node'''
    global nodes
    sum_R = 0
    for node in neighbors:
        sum_R += nodes[node][0]
    return int(sum_R/len(neighbors))


def remove_suicides(G):
    '''Remove nodes who surpass the suicide threshold
    
    Returns the number of suicides
    '''
    global nodes
    old_size = len(nodes)
    to_delete = [] # List of row indices to delete
    for i, node in enumerate(nodes):
        if node[0]/(node[0] + node[1]) >= S_THRESHOLD:
            # Set node for removal from nodes array
            completed = random.uniform(0,1)
            if completed > 0.5:
                to_delete.append(i)
                # For now, connect all of the node's neighbors to each other
                # With a 10% chance of connection
                neighbors = list(G[i])
                for j, node_1 in enumerate(neighbors[:-1]):
                    for node_2 in neighbors[j+1:]:
                        if random.random() < 0.1:
                            G.add_edge(node_1, node_2)
                if not nx.is_connected(G):
                    # We've made the graph disconnected.
                    # For now, fix this by adding connections between neighbors
                    # until it is connected
                    done = False
                    for j, node_1 in enumerate(neighbors[:-1]):
                        if done:
                            break
                        for node_2 in neighbors[j+1:]:
                            G.add_edge(node_1, node_2)
                            done = nx.is_connected(G)
                            if done:
                                break
                G.remove_node(i)
            else:
                # Suidice attempt occurred, but did not result in death
                nodes[i][0] = nodes[i][0] * 0.9 # Remove some red balls
    nodes = np.delete(nodes, to_delete, axis=0)
    # Relabel the nodes so the graph's labels match up again with the nodes array indices
    old_names = [x for x in range(old_size) if x not in to_delete]
    new_names = range(len(nodes))
    mapping = {old: new for old, new in zip(old_names, new_names)}
    nx.relabel_nodes(G, mapping, copy=False)
    return len(to_delete)


# ========================= Mitigation Methods =========================

def random_mitigation(budget):
    '''Add black balls randomly to nodes.
    Budget is an absolute number of balls.'''
    global nodes
    # Eventually this number gets very large.  We need to bound the number of steps.
    step = 1
    if budget > len(nodes) * 10:
        step = budget // (len(nodes) * 5)
    while budget > 0:
        i = random.randrange(len(nodes))
        nodes[i][1] += step
        budget -= step

def random_mitigation_percentage(budget_percentage):
    '''Add black balls randomly to nodes.
    Budget is a percentage of the total number of balls in the system'''
    global nodes
    # We can't just sum all of the nodes and then multiply by the percentage,
    # the number gets too large.
    budget = 0
    for _, node in enumerate(nodes):
        budget += (node[0] + node[1]) * budget_percentage
    random_mitigation(budget)

# ==================== Update Function and Helpers ====================

def calculate_proportions(print_out=False):
    global nodes
    prop = []
    for node in nodes:
        temp = 100*node[0]/(node[0] + node[1])
        prop.append(temp)
        if (print_out):
            print(f"{temp:2.0f}%")
    return prop


def updateFunc(step, G, pos):
    global nodes
    print()
    ## Apply mitigations
    random_mitigation_percentage(0.01)
    ## Run main simulation step
    plt.clf()   # Without this, the colorbars act all weird
    new_nodes = nodes.copy() # Careful, copy the array!
    for i in range(len(nodes)):
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
    nodes = new_nodes
    ## Remove suicides
    num_suicides = remove_suicides(G)
    ## Calculate metrics
    metrics.append(calculte_metrics(G))
    ## Record Suicides
    metrics[-1]["suicides"] = num_suicides
    ## Print
    print(nodes)
    prop_current = calculate_proportions()
    show_network(G, pos, prop_current)
    plt.title(f"Step {step+1}")
    #plt.show()

# ============================== Main Loop ==============================

def main():
    '''Main setup and loop'''
    global nodes
    # Setup
    G = ba_graph()
    #G = clique_graph()
    #G = houses_graph()
    pos = nx.spring_layout(G)
    #init_nodes_profiles()
    init_nodes()
    # Check
    check_setup(G)

    # Show network Information
    print(nx.info(G))
    print(f"Density: {nx.density(G)}")
    print(f"Diameter: {nx.diameter(G)}")
    # WARNING - This next one is very slow for large graphs
    print("Average node connectivity: ", nx.average_node_connectivity(G))

    print()

    # Calculate initial proportions
    # Don't print this time
    prop_before = calculate_proportions()

    # Show Network Before
    show_network(G, pos, prop_before)
    plt.title("Before")
    plt.show()

    # Loop
    fig = plt.figure()
    animator = ani.FuncAnimation(fig, updateFunc, fargs=(G,pos), interval=100, frames=NUM_TIME_STEPS, repeat=False)
    plt.show()

    # Print Result
    print()
    print("Proportion of Red Before")
    for val in prop_before:
        print(f"{val:2.0f}%")

    print()

    print("Proportion of Red After")
    prop_after = calculate_proportions(print_out=True)

    # Show Network After
    show_network(G, pos, prop_after)
    plt.title("After")
    plt.show()

    # Show metrics
    plt.plot([metrics_dict["network susceptibility"] for metrics_dict in metrics])
    plt.title("Network Susceptibility")
    plt.show()

    plt.plot([metrics_dict["network exposure"] for metrics_dict in metrics])
    plt.title("Network Exposure")
    plt.show()

    plt.plot([metrics_dict["suicides"] for metrics_dict in metrics])
    plt.title("Suicides by time step")
    plt.show()
    total_suicides = sum([metrics_dict["suicides"] for metrics_dict in metrics])
    print(f"Average suicide rate: {total_suicides/NUM_TIME_STEPS:.2f} per step")

    plt.plot([metrics_dict["suicial thoughts"] for metrics_dict in metrics])
    plt.title("Number of People with Suicidal Thoughts by time step")
    plt.show()

if __name__ == "__main__":
    main()
