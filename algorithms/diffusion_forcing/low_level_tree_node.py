import numpy as np


class LowLevelTreeNode():
    """Denoising-level tree node of the MCTD node planner (one tree per plan segment).

    A node at depth d holds the segment denoised to the d-th stage of the schedule. Its child slots
    carry the guidance values (``slots``, each value repeated for independent samples); a claimed
    child is denoised one stage further with its slot's guidance and scored by a jumpy rollout.
    ``value`` is the max-backed-up score, ``virtual`` counts the selections within the current
    parallel batch (redundancy-aware selection, Fast-MCTD Eq. 5).
    """

    def __init__(self, depth, slots, parent=None, guidance=None):
        self.depth = depth
        self.slots = slots
        self.parent = parent
        self.guidance = guidance
        self.children = []
        self.plan = None  # normalized partially denoised plan (t, 1, c), set after its expansion
        self.score = None  # rollout score, set after its expansion
        self.info = None  # scoring status of the rollout (Achieved, NotReached, ..., NoValidCandidate)
        self.value = None
        self.visits = 0
        self.virtual = 0

    def select(self, terminal_depth, exp_weight, virtual_visit_weight):
        """A node with an unclaimed child slot, descending by UCT with redundancy-aware selection."""
        if self.depth >= terminal_depth:
            return None
        if len(self.children) < len(self.slots):
            return self
        children = [c for c in self.children if c.value is not None]  # skip children still in the batch
        visits = [c.visits + virtual_visit_weight * c.virtual for c in children]
        total = sum(visits)
        scores = [c.value + exp_weight * np.sqrt(np.log(1e-6 + max(1, total)) / (1e-6 + v))
                  for c, v in zip(children, visits)]
        for k in sorted(range(len(children)), key=lambda k: -scores[k]):
            selected = children[k].select(terminal_depth, exp_weight, virtual_visit_weight)
            if selected is not None:
                return selected
        return None

    def claim_child(self):
        """Create the child of the next unclaimed slot and add a virtual visit on the selected path."""
        child = LowLevelTreeNode(self.depth + 1, self.slots, parent=self, guidance=self.slots[len(self.children)])
        self.children.append(child)
        node = self
        while node is not None:
            node.virtual += 1
            node = node.parent
        return child

    def backpropagate(self):
        """Delayed update of a scored child: its score becomes its value, max backup to the root."""
        self.value, self.visits = self.score, 1
        node = self.parent
        while node is not None:
            node.virtual -= 1
            node.visits += 1
            node.value = max(c.value for c in node.children if c.value is not None)
            node = node.parent
