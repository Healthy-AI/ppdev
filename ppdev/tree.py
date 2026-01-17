import numpy as np
from sklearn.tree import _tree

TREE_UNDEFINED = _tree.TREE_UNDEFINED
TREE_LEAF = _tree.TREE_LEAF
NODE_DTYPE = _tree.NODE_DTYPE


def get_all_nodes(root_node):
    nodes = []
    node_stack = [root_node]
    while node_stack:
        node = node_stack.pop()
        nodes.append(node)
        if node.right_subtree is not None:
            node_stack.append(node.right_subtree)
        if node.left_subtree is not None:
            node_stack.append(node.left_subtree)
    return nodes


class ExtendedTree:
    def __init__(
        self,
        n_features,
        n_classes,
        tree=None,
        root_node=None,
    ):
        if tree is None and root_node is None:
            raise ValueError("Either tree or root_node must be provided.")

        if tree is not None:
            self.tree = tree
            self.nodes = None
        else:
            self.tree = convert_to_sklearn_tree(
                root_node, n_features, n_classes
            )
            self.nodes = get_all_nodes(root_node)

        self.capacity = self.tree.capacity
        self.max_n_classes = self.tree.max_n_classes
        self.n_classes = self.tree.n_classes
        self.n_features = self.tree.n_features
        self.n_leaves = self.tree.n_leaves
        self.n_outputs = self.tree.n_outputs

        state = self.tree.__getstate__()
        state["node_array"] = state.pop("nodes")
        state["value_array"] = state.pop("values")
        self.__dict__.update(state)

    @property
    def children_left(self):
        return self.node_array["left_child"][:self.node_count]

    @property
    def children_right(self):
        return self.node_array["right_child"][:self.node_count]

    @property
    def feature(self):
        return self.node_array["feature"][:self.node_count]

    @property
    def threshold(self):
        return self.node_array["threshold"][:self.node_count]

    @property
    def impurity(self):
        return self.node_array["impurity"][:self.node_count]

    @property
    def n_node_samples(self):
        return self.node_array["n_node_samples"][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        return self.node_array["weighted_n_node_samples"][:self.node_count]

    @property
    def missing_go_to_left(self):
        return self.node_array["missing_go_to_left"][:self.node_count]

    @property
    def value(self):
        return self.value_array[:self.node_count]

    def store_outcomes(self, X, y, o):
        leaves = self.apply(X)
        self.outcome_per_class = np.full((self.node_count, self.max_n_classes), np.nan)
        for leaf in np.unique(leaves):
            for c in range(self.max_n_classes):
                mask = (leaves == leaf) & (y == c) & ~np.isnan(o)
                if np.any(mask):
                    self.outcome_per_class[leaf, c] = np.mean(o[mask])

    def predict(self, X):
        return self.tree.predict(X)

    def apply(self, X):
        return self.tree.apply(X)
    
    def decision_path(self, X):
        return self.tree.decision_path(X)

    def compute_node_depths(self):
        return self.tree.compute_node_depths()

    def compute_feature_importances(self, normalize=True):
        return self.tree.compute_feature_importances(normalize)


def convert_to_sklearn_tree(root_node, n_features, n_classes):
    nodes = get_all_nodes(root_node)

    max_depth = max(node.depth for node in nodes)
    node_count = len(nodes)

    values = []
    for node in nodes:
        if np.isscalar(node.value):
            value = [node.value] + [-1] * (n_classes - 1)
            values.append(value)
        else:
            values.append(node.value)
    values = np.array(values).reshape(-1, 1, n_classes)

    nodes = [
        (
            node.left_child,
            node.right_child,
            node.feature,
            node.threshold,
            node.impurity,
            node.n_node_samples,
            node.weighted_n_node_samples,
            node.missing_go_to_left,
        )
        for node in nodes
    ]
    nodes = np.array(nodes, dtype=NODE_DTYPE)

    tree = _tree.Tree(n_features, np.atleast_1d(n_classes), 1)
    tree.__setstate__(
        {
            "max_depth": max_depth,
            "node_count": node_count,
            "nodes": nodes,
            "values": values,
        }
    )

    return tree
