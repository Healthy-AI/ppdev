import re
from numbers import Integral

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_categorical_dtype, is_bool_dtype

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _criterion,
)
from sklearn.tree._reingold_tilford import Tree
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import validate_params, Interval, StrOptions
from sklearn.tree._export import _MPLTreeExporter


class TreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
        node_ids_to_include=None,
    ):
        self.node_ids_to_include = node_ids_to_include
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )
    
    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if (
            self.node_ids_to_include is not None
            and node_id not in self.node_ids_to_include
            and not self.node_ids
        ):
            name = self.node_to_str(et, node_id, criterion=criterion)
            left = et.children_left[node_id]
            right = et.children_right[node_id]
            if left != -1 and right != -1:
                splits = name.split("\n")
                splits[0] = "null"
                name = "\n".join(splits)
            return Tree(name, node_id)
        return super()._make_tree(node_id, et, criterion, depth)

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                if node.tree.label.startswith("null"):
                    kwargs["bbox"]["fc"] = "lightgrey"
                    label = node.tree.label.replace("null", "(...)")
                    ax.annotate(label, xy_parent, xy, **kwargs)
                else:
                    ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            if not node.tree.label.startswith("null"):
                for child in node.children:
                    self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "lightgrey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


@validate_params(
    {
        "decision_tree": [
            DecisionTreeClassifier,
            DecisionTreeRegressor,
        ],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rounded": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "ax": "no_validation",  # Delegate validation to matplotlib
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    label_mapper=None,
    formatter=None,
    annotate_arrows=False,
    revert_true_false=False,
    inverse_transformer=None,
    node_ids_to_include=None,
    x_adjustment=(3, 3),
    y_adjustment=(2, 2),
):
    check_is_fitted(decision_tree)

    exporter = TreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        node_ids_to_include=node_ids_to_include,
    )
    annotations = exporter.export(decision_tree, ax=ax)

    if node_ids or label != "all":
        return annotations

    if ax is None:
        ax = plt.gca()

    if label_mapper is None:
        label_mapper = {}

    criterion = decision_tree.criterion
    if isinstance(criterion, _criterion.FriedmanMSE):
        criterion = "friedman_mse"
    elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
        criterion = "squared_error"
    elif not isinstance(criterion, str):
        criterion = "impurity"

    renderer = ax.figure.canvas.get_renderer()
    for annotation in annotations:
        text = annotation.get_text()
        if text.startswith(criterion) or text.startswith("samples"):
            # Leaf node
            if formatter is not None:
                if impurity:
                    i, s, v = text.split("\n")
                    i, s, v = formatter(i, s, v)
                    text = "\n".join([s, i, v])
                else:
                    s, v = text.split("\n")
                    s, v = formatter(s, v)
                    text = "\n".join([s, v])
        elif text.startswith("True"):
            text = ""
        elif text.endswith("False"):
            text = ""
        elif text.startswith("\n"):
            # (...)
            pass
        else:
            # Inner node
            l = text.split("\n")[0]
            if l in label_mapper:
                l = label_mapper[l]
            elif re.match(r".*?\s*<=\s*-?\w+", l):
                l1, l2 = l.split(" <= ")
                l2 = float(l2)
                if inverse_transformer is not None:
                    try:
                        l2 = inverse_transformer(l1, l2)
                    except ValueError:
                        pass
                l1 = label_mapper.get(l1, l1)
                l = l1 + r" $\leq$ " + "{:.{prec}f}".format(l2, prec=precision)
            if impurity:
                try:
                    _l, i, s, v = text.split("\n")
                except ValueError:
                    _l, i, s, v1, v2 = text.split("\n")
                    v = f"{v1}, {v2}"
                if formatter is not None:
                    try:
                        i, s, v = formatter(i, s, v)
                    except ValueError:
                        i, s, v1, v2 = formatter(i, s, v)
                        v = f"{v1}, {v2}"
                text = "\n".join([l, i, s, v])
            else:
                _l, s, v = text.split("\n")
                if formatter is not None:
                    s, v = formatter(s, v)
                text = "\n".join([l, s, v])
        annotation.set_text(text)
        annotation.set(ha="center")
        annotation.draw(renderer)

    if annotate_arrows:
        kwargs = dict(ha="center", va="center", fontsize=fontsize)
        x0, y0 = annotations[0].get_position()
        x1, y1 = annotations[1].get_position()
        t = "True" if not revert_true_false else "False"
        f = "False" if not revert_true_false else "True"
        x1_adj, x2_adj = x_adjustment
        y1_adj, y2_adj = y_adjustment
        ax.annotate(t, (x1 + (x0-x1) / x1_adj, y0 - (y0-y1) / y1_adj), **kwargs)
        ax.annotate(f, (x0 + 2 * (x0-x1) / x2_adj, y0 - (y0-y1) / y2_adj), **kwargs)


def describe_categorical(D):
    assert (D.apply(is_categorical_dtype) | D.apply(is_bool_dtype)).all()

    index_tuples = []
    out = {"Counts": [], "Proportions": []}

    for v in D:
        s = D[v]

        if is_bool_dtype(s):
            s = s.astype("category")
            s = s.cat.rename_categories({True: "yes", False: "no"})

        table = pd.Categorical(s).describe()

        # Exclude NaNs when computing proportions.
        N = table.counts.values[:-1].sum() if -1 in table.index.codes \
            else table.counts.values.sum()
        proportion = [round(100 * x, 1) for x in table.counts / N]
        table.insert(1, "proportion", proportion)

        try:
            from ppdev.data.variables import COREVITAS_DATA
            categories = COREVITAS_DATA.variables[s.name].pop("categories", None)
        except ModuleNotFoundError:
            categories = None
        except KeyError:
            categories = None

        if categories is not None:
            table.index = table.index.rename_categories(categories)

        for c in table.index:
            index_tuples.append((v, N, c))
        out["Counts"].extend(table.counts)
        out["Proportions"].extend(table.proportion)

    index = pd.MultiIndex.from_tuples(
        index_tuples, names=["Variable", "No. samples", "Value"]
    )
    return pd.DataFrame(out, index=index)


def describe_numerical(D):
    table = D.describe().T
    table.rename(columns={"count": "No. samples"}, inplace=True)
    return table.drop(
        columns=["mean", "std", "min", "max"]
    )


def display_dataframe(
    df,
    caption=None,
    new_index=None,
    new_columns=None,
    hide_index=False,
    precision=2,
):
    def set_style(styler):
        if caption is not None:
            styler.set_caption(caption)
        if new_index is not None:
            styler.relabel_index(new_index, axis=0)
        if new_columns is not None:
            styler.relabel_index(new_columns, axis=1)
        if hide_index:
            styler.hide(axis="index")
        styler.format(precision=precision)
        return styler

    display_everything = (
        "display.max_rows", None,
        "display.max_columns", None,
    )

    with pd.option_context(*display_everything):
        return df.style.pipe(set_style)
