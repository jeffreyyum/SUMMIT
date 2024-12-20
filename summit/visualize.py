import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.drawing.nx_agraph import graphviz_layout


def visualize_chunk_tree_improved(final_chunks):
    """
    Improved visualization of the tree structure of chunks with better root emphasis and hierarchy.

    Args:
    - final_chunks (List[Chunk]): A list of chunks, including level 0 and tree-generated chunks.

    Returns:
    - None: Displays the tree visualization.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Assign colors to entities
    entity_colors = {}
    for chunk in final_chunks:
        if chunk.entity and chunk.entity not in entity_colors:
            # Assign a random color to each entity
            entity_colors[chunk.entity] = (
                random.random(),
                random.random(),
                random.random(),
            )

    # Add nodes and edges
    for chunk in final_chunks:
        # Determine node label and attributes
        if chunk.level == 0 and chunk.entity:
            # Root nodes: Highlight with entity name
            node_label = f"{chunk.entity} (Root)"
            node_color = "gold"  # Unique color for root nodes
            node_size = 4000  # Larger size for root nodes
            font_size = 14
        else:
            # Other nodes: Use chunk text
            node_label = chunk.text[:20] + "..."  # Truncated label for readability
            node_color = entity_colors.get(chunk.entity, (0.7, 0.7, 0.7))
            node_size = 2000
            font_size = 10

        # Add the node to the graph
        G.add_node(chunk.id, label=node_label, color=node_color, size=node_size, font_size=font_size)

        # Add edges to children
        for child in chunk.children:
            G.add_edge(chunk.id, child.id)

    # Extract attributes for visualization
    node_labels = nx.get_node_attributes(G, "label")
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    node_sizes = [G.nodes[node]["size"] for node in G.nodes]

    # Use graphviz layout for better hierarchical visualization
    try:
        pos = graphviz_layout(G, prog="dot")  # Hierarchical layout
    except ImportError:
        print("Graphviz layout not available, using spring layout.")
        pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        labels=node_labels,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        font_color="black",
        font_weight="bold",
        edge_color="gray",
        arrows=True,
        arrowsize=15,  # Larger arrows for clarity
    )

    # Add a title
    plt.title("Improved Chunk Tree Visualization", fontsize=18)
    plt.show()

