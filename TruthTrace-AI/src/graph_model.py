import networkx as nx
from typing import List, Dict, Any
from .data_model import Message

class PropagationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, messages: List[Message]):
        """Constructs a directed graph from the message chain."""
        self.graph.clear()
        for msg in messages:
            self.graph.add_node(msg.id, text=msg.text, timestamp=msg.timestamp)
            if msg.parent_id:
                # Edge points from parent (sender) to child (receiver)
                self.graph.add_edge(msg.parent_id, msg.id)

    def get_node_depth(self, node_id: str) -> int:
        """Calculates the hop distance from the original source message (M0)."""
        # Find the root node(s) - nodes with no incoming edges
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        if not roots:
            return 0
        
        try:
            return nx.shortest_path_length(self.graph, source=roots[0], target=node_id)
        except nx.NetworkXNoPath:
            return 0

    def get_max_depth(self) -> int:
        """Returns the length of the longest propagation chain."""
        if nx.is_directed_acyclic_graph(self.graph):
            return nx.dag_longest_path_length(self.graph)
        return 0