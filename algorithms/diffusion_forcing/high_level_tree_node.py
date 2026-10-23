import math

import numpy as np


class HighLevelTreeNode():
    """Plan-level tree node: a node at depth d holds d stitched plans.

    Each child slot carries a guidance set (the meta-action) used to sample the
    child's plan. ``local_value`` accumulates the expansion scores along the path;
    ``value`` is the UCT value, set by simulation and max-backed-up from children.
    """

    def __init__(self, name, depth, parent_node, children_guidance_sets, plan_history, guidance_set=None, local_value=0):
        self.name = name
        self.depth = depth
        self._parent_node = parent_node
        self._children_guidance_sets = children_guidance_sets
        self.plan_history = [plan.detach().cpu() for plan in plan_history]
        self.guidance_set = guidance_set
        # The maximum number of children nodes is same to the number of children guidance sets
        self._children_nodes = [
            {'guidance_set': self._children_guidance_sets[i], 'node': None}
            for i in range(len(self._children_guidance_sets))
        ]

        self.visit_count = 0
        self.virtual_visit_count = 0  # selections within the current parallel batch (Fast-MCTD RAS)
        self.local_value = local_value
        self.value = None
        self.value_estimation_plan = None
        self.local_name = None
        self.invalid = False

    def reject(self):
        """Keep the attempted child for accounting, but never expand it again."""
        self.invalid = True
        self.value = 0.0
        if self._parent_node is not None:
            self._parent_node.backpropagate()

    def plan_length(self):
        return sum(len(plan) for plan in self.plan_history)

    def select_expandable(self, terminal_depth, exp_weight=math.sqrt(2), virtual_visit_weight=0.0,
                          max_plan_length=None):
        """Select by UCT among branches that still have a usable expansion.

        Children created earlier in the same parallel batch have no value yet and are
        skipped. ``virtual_visit_weight`` adds the batch's own selections to the visit
        counts (redundancy-aware selection, Fast-MCTD Eq. 5). With ``max_plan_length``
        a node whose stitched plans already reach that many points is terminal.
        """
        if self.invalid or self.depth >= terminal_depth:
            return None
        if max_plan_length is not None and self.plan_length() >= max_plan_length:
            return None
        if self.is_expandable():
            return self
        children = [entry['node'] for entry in self._children_nodes if entry['node'].value is not None]
        visits = [node.visit_count + virtual_visit_weight * node.virtual_visit_count for node in children]
        total_visits = sum(visits)
        scores = [node.value + exp_weight * np.sqrt(
            np.log(1e-6 + max(1, total_visits)) / (1e-6 + visit))
            for node, visit in zip(children, visits)]
        for index in sorted(range(len(children)), key=lambda i: -scores[i]):
            selected = children[index].select_expandable(
                terminal_depth, exp_weight, virtual_visit_weight, max_plan_length)
            if selected is not None:
                return selected
        return None

    def add_virtual_visit(self):
        node = self
        while node is not None:
            node.virtual_visit_count += 1
            node = node._parent_node

    def clear_virtual_visits(self):
        node = self
        while node is not None:
            node.virtual_visit_count = 0
            node = node._parent_node

    def is_expandable(self):
        if self.invalid:
            return False
        for child_node in self._children_nodes:
            if child_node["node"] is None:
                return True
        return False

    def set_value(self, value):
        self.value = value

    def add_local_value(self, local_value):
        self.local_value += local_value

    def set_local_name(self, local_name):
        self.local_name = local_name

    def set_plan_history(self, plan_history):
        self.plan_history.append(plan_history.detach().cpu())

    def set_value_estimation_plan(self, value_estimation_plan):
        self.value_estimation_plan = value_estimation_plan.detach().cpu()

    def get_start(self):
        if len(self.plan_history) == 0:
            return None
        else:
            # the last state of the last plan
            plan = self.plan_history[-1]
            return plan[-1]

    def expand(self):
        if self.invalid:
            raise ValueError('Cannot expand a rejected node')
        remaining_children_indices = [
            i for i in range(len(self._children_nodes)) if self._children_nodes[i]['node'] is None
        ]
        selected_index = np.random.choice(remaining_children_indices)
        # Create a new child node
        self._children_nodes[selected_index]['node'] = HighLevelTreeNode(
            name=self.name + f'-{selected_index}',
            depth=self.depth + 1,
            parent_node=self,
            children_guidance_sets=self._children_guidance_sets,
            plan_history=self.plan_history.copy(),
            guidance_set=self._children_nodes[selected_index]['guidance_set'],
            local_value=self.local_value
        )
        return self._children_nodes[selected_index]['node']

    def backpropagate(self):
        self.visit_count += 1
        # Children created earlier in the same parallel batch have no value yet.
        values = [child_node["node"].value for child_node in self._children_nodes
                  if child_node["node"] is not None and child_node["node"].value is not None]
        if values:
            self.value = max(values)
        if self._parent_node is not None:
            self._parent_node.backpropagate()
