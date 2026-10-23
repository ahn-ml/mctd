import math

import numpy as np


class CubeTreeNode():
    """Object-level tree node of the cube tasks: a node moves one cube (``node_spec`` = [cube index, target]).

    ``plan`` is the plan of that move from the parent's end state; ``value`` is the share of the cubes at
    their goal after it (-inf prunes the branch), max-backed-up from the children.
    """

    def __init__(self, name, depth, parent_node, node_spec=None):
        self.name = name
        self.depth = depth
        self._parent_node = parent_node
        self._node_spec = node_spec
        self._children_nodes = []
        self._children_node_specs = []

        self._value = None
        self._plan = None
        self._visit_count = 0
        self._virtual_visit_count = 0  # selections within the current parallel batch

    def __str__(self):
        return f"Name: {self.name}, Depth: {self.depth}, Node Spec: {self._node_spec}, Value: {self._value}, Visit Count: {self._visit_count}"

    def is_root_node(self):
        return self._parent_node is None

    def is_expandable(self):
        return len(self._children_nodes) < len(self._children_node_specs)

    def set_value(self, value):
        self._value = value

    def set_plan(self, plan):
        self._plan = plan

    def set_children_node_specs(self, children_node_specs):
        self._children_node_specs = children_node_specs

    def get_parent_node(self):
        return self._parent_node

    def get_value(self):
        return self._value

    def get_plan(self):
        return self._plan

    def get_visit_count(self):
        return self._visit_count

    def get_node_spec(self):
        return self._node_spec

    def select_expandable(self, exp_weight=math.sqrt(2), virtual_visit_weight=1.0):
        """Select by UCT among the branches that still have a node to expand; None if there is none.

        Pruned children (-inf) and children created earlier in the same parallel batch (no value yet) are
        skipped. ``virtual_visit_weight`` adds the batch's own selections to the visit counts (redundancy-aware
        selection, Fast-MCTD); a branch without an expandable node falls back to the next best one.
        """
        if self._value == -np.inf:
            return None
        if self.is_expandable():
            return self
        children = [child_node for child_node in self._children_nodes
                    if child_node.get_value() is not None and child_node.get_value() != -np.inf]
        visits = [child_node.get_visit_count() + virtual_visit_weight * child_node._virtual_visit_count
                  for child_node in children]
        total_visits = sum(visits)
        scores = [child_node.get_value() + exp_weight * np.sqrt(np.log(1e-6 + max(1, total_visits)) / (1e-6 + visit))
                  for child_node, visit in zip(children, visits)]
        for index in sorted(range(len(children)), key=lambda i: -scores[i]):
            selected = children[index].select_expandable(exp_weight, virtual_visit_weight)
            if selected is not None:
                return selected
        return None

    def expand(self):
        """Create the child of the next child spec."""
        next_child_node_spec = self._children_node_specs[len(self._children_nodes)]
        self._children_nodes.append(CubeTreeNode(
            name=f"{self.name}-{len(self._children_nodes)}",
            depth=self.depth + 1,
            parent_node=self,
            node_spec=next_child_node_spec
        ))
        return self._children_nodes[-1]

    def add_virtual_visit(self):
        node = self
        while node is not None:
            node._virtual_visit_count += 1
            node = node._parent_node

    def clear_virtual_visits(self):
        node = self
        while node is not None:
            node._virtual_visit_count = 0
            node = node._parent_node

    def backpropagate(self):
        self._visit_count += 1
        values = [child_node.get_value() for child_node in self._children_nodes if child_node.get_value() is not None]
        if not self.is_expandable():  # all children nodes are expanded
            self._value = max(values)
        else:  # not all children nodes are expanded
            self._value = max(self._value, max(values))
        if self._parent_node is not None:
            self._parent_node.backpropagate()
