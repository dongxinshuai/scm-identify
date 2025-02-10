# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:00:33 2023

@author: AmberSun
"""

import numpy as np


def extract_graph_edges(text):
    lines = text.strip().split("\n")
    start_idx = lines.index("Graph Edges:") + 1  # +1 to skip the "Graph Edges:" line itself
    
    # Collect edges
    edges = []
    for line in lines[start_idx:]:
        if line.strip() == "":  # Stop at an empty line
            break
        edges.append(line.split('. ')[1])  # Skip the numbering (e.g., "1. ")
    
    return edges


def extract_graph_nodes(text):
    # Split the text into lines
    lines = text.strip().split("\n")
    
    # Find the line containing "Graph Nodes:"
    idx = lines.index("Graph Nodes:")
    
    # The next line contains the nodes separated by ';'
    nodes = lines[idx + 1].split(";")
    
    return nodes


def generate_adjacency_matrix_ordered(edges_list, nodes_order):
    """
    Generate an adjacency matrix based on the provided edges and node order.
    
    Parameters:
        edges_list (list): List of edges in the format "source relation target".
        nodes_order (list): List of node names in the desired order.
    
    Returns:
        numpy.ndarray: Adjacency matrix.
    """
    node_to_index = {node: idx for idx, node in enumerate(nodes_order)}
    
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((len(nodes_order), len(nodes_order)), dtype=int)
    
    # Update adjacency matrix based on edges
    for edge in edges_list:
        source, relation, target = edge.split(' ')
        if relation == '-->':
            adj_matrix[node_to_index[source], node_to_index[target]] = -1
            adj_matrix[node_to_index[target], node_to_index[source]] = 1
        elif relation == '<-->':
            adj_matrix[node_to_index[source], node_to_index[target]] = 1
            adj_matrix[node_to_index[target], node_to_index[source]] = 1
        elif relation == '---':
            adj_matrix[node_to_index[source], node_to_index[target]] = -1
            adj_matrix[node_to_index[target], node_to_index[source]] = -1

    return adj_matrix