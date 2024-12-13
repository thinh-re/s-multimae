from itertools import product
import os
from pprint import pprint
import networkx as nx
from PIL import Image
import numpy as np
import networkx as nx


def shortest_cycle_through_all_nodes(G: nx.Graph):
    # Initialize the shortest cycle length with infinity
    shortest_cycle_length = float("inf")
    shortest_cycle = None

    i = 0

    # Iterate through all permutations of the nodes
    for cycle in nx.simple_cycles(G):
        # Check if the cycle goes through all nodes
        if set(cycle) == set(G.nodes()):
            i += 1
            # Calculate the length of the cycle
            cycle_length = sum(
                G[u][v]["weight"] for u, v in zip(cycle, cycle[1:] + cycle[:1])
            )
            print(cycle, cycle_length)
            # Update the shortest cycle if this one is shorter
            if cycle_length < shortest_cycle_length:
                shortest_cycle_length = cycle_length
                shortest_cycle = cycle

    print("num cycles", i)

    return shortest_cycle, shortest_cycle_length


if __name__ == "__main__":

    # Create a sample graph
    G = nx.Graph()
    # G.add_edge("A", "B", weight=2)
    # G.add_edge("B", "C", weight=3)
    # G.add_edge("C", "A", weight=1)  # forms a cycle

    files = [
        f"data/samples/3/{f}"
        for f in os.listdir("data/samples/3")
        if f.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]
    images = [Image.open(file).convert("L") for file in files]

    distance_list = []
    for i, j in product(range(len(images)), repeat=2):
        if i != j:
            img1 = np.array(images[i])
            img2 = np.array(images[j])
            intersection = np.logical_and(img1, img2).sum()
            union = np.logical_or(img1, img2).sum()
            distance = 1.0 - float(intersection / union)
            G.add_edge(i, j, weight=distance)
            distance_list.append([i, j, distance])

    pprint(distance_list)

    # Example usage
    cycle, length = shortest_cycle_through_all_nodes(G)
    print(G)
    print("Shortest cycle:", cycle)
    print("Cycle length:", length)
