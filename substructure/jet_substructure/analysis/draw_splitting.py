#!/usr/bin/env python3

""" Functionality for drawing and exploring splittings.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import attr

from jet_substructure.base import substructure_methods


if TYPE_CHECKING:
    import networkx

logger = logging.getLogger(__name__)


@attr.s
class EdgeFromSubjet:
    edge: Tuple[Union[str, int], Union[str, int]] = attr.ib()
    subjet_index: int = attr.ib()
    subjet_pt: float = attr.ib()
    part_of_iterative_splitting: bool = attr.ib()


@attr.s
class EdgeFromSplitting:
    edge: Tuple[Union[str, int], Union[str, int]] = attr.ib()
    part_of_iterative_splitting: bool = attr.ib()


def splittings_graph(  # noqa: C901
    jet: substructure_methods.SubstructureJet,
    path: Path,
    filename: Optional[str] = None,
    show_subjet_pt: bool = False,
    selected_splitting_index: int = -1,
) -> "networkx.DiGraph":
    """Draw a splitting graph for a given jet.

    In the graph, inner nodes represent splittings, and lines represent subjets. Outer nodes (ie those with no
    nodes leaving them) represent particles. The iterative splitting nodes are shown in green, while the iterative
    splitting subjets are shown in blue. Each node is labeled with the kt of the splitting. Each subjet can be labeled
    with the subjet pt based on a four vector sum of the constituents (if `show_sum_of_constituent_pts` is enabled).

    Args:
        jet: Jet to be plotted.
        path: Path to where the plot should be stored.
        filename: Filename under which the plot should be stored.
        show_subjet_pt: If True, we will label the subjets their pt calculated via their constituents.
            This can be used to very that we've properly following the hardest splitting. Default: False.
        select_splitting_index: Highlight the specified splitting. Default: -1, in which case, no splitting
            will be highlighted.
    Returns:
        The directed graph representing the splitting. A plot of the graph is also stored at the provided path and filename.
    """
    # Import the necessary packages. We make this a lazy import because we don't want a hard dependency.
    try:
        import networkx as nx

        # pygraphviz is required for drawing the graph. We only use it indirectly, but importing it ensures
        # that it's available.
        import pygraphviz  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "Failed to import network or pygraphviz. Please install them."
            " You also need to install graphviz externally (for example, via brew)."
        )
        raise exc

    # Validation
    path = Path(path)
    if filename is None:
        filename = f"splitting_graph_jetPt_{jet.jet_pt:.1f}GeV"

    logger.info(
        f"Drawing jet with pt={jet.jet_pt:g} GeV/c, and {len(jet.splittings)} splittings ({len(jet.splittings.iterative_splittings(jet.subjets))} iterative)"
    )

    # We want a directed graph showing the splitting.
    graph = nx.DiGraph()
    # Nodes represent each splitting
    # We create the notes by specifying the edges, which then will create the necessary nodes.
    # NOTE: We can't just use enumerate because we have a directed graph and we need to have the parent index first.
    splittings_indices = list(zip(jet.splittings.parent_index, range(len(jet.splittings.parent_index))))
    splittings: List[Tuple[str, str]] = []
    # Add prefix to the splittings nodes so we can differentiate them from the subjets
    for v in splittings_indices:
        splittings.append((f"s{v[0]}", f"s{v[1]}"))
    # Add edges from the edges (and implicitly, the nodes that are necessary for those edges)
    graph.add_edges_from(splittings)
    # Remove the -1 node, as it's a fake node to indicate the root (origin) splitting.
    graph.remove_node("s-1")
    # Associate the splittings index with the nodes so we can get the splitting later.
    # We don't store a reference to the object directly because pygraphviz requires it to be a str.
    # It might work for networkx, but it's not worth changing now.
    for i, splitting in enumerate(jet.splittings):
        n = graph.nodes[f"s{i}"]
        n["splitting_index"] = i

    # Add subjets edges
    subjets: List[Tuple[str, int]] = []
    subjets_indices = list(
        zip(jet.subjets, jet.subjets.parent_splitting_index, range(len(jet.subjets.parent_splitting_index)))
    )
    for subjet, parent_splitting_index, subjet_index in subjets_indices:
        # Need to label the splittings with the prefix.
        subjets.append((f"s{parent_splitting_index}", subjet_index))
    graph.add_edges_from(subjets)
    # Associated the subjets with the edges.
    for subjet, parent_splitting_index, subjet_index in subjets_indices:
        e = graph.edges[(f"s{parent_splitting_index}", subjet_index)]
        e["subjet_index"] = subjet_index

    # Now that all necessary parts of the graph have been created, we need to remove any duplicated edges
    # First, We need to be able to identify all iterative splitting nodes. Critically, this also must include
    # the splitting node prefix.
    iterative_splitting_indices_with_prefix = [f"s{v}" for v in jet.subjets.iterative_splitting_index]

    # Now, we can merge subjets edges into the edges created by the splittings.
    nodes_to_remove: List[Union[str, int]] = []
    for n in graph.nodes():
        # We are only interested in splitting nodes.
        if "s" not in str(n):
            continue
        # There should only be two edges leaving the node.
        edges_labels = graph.out_edges(n)
        # Sanity check
        if len(edges_labels) > 4:
            raise ValueError(f"Too many edges for node {n} (len: {len(edges_labels)}). Something has gone wrong!")
        # Only remove edges if necessary.
        if len(edges_labels) > 2:
            # To determine which edges to remove, we need to identify which edges correspond to subjets
            # and which edges are from the splittings. We want to effectively merge the subjet edges into
            # the splitting edges.
            subjet_edges: List[EdgeFromSubjet] = []
            splitting_edges: List[EdgeFromSplitting] = []
            for e in edges_labels:
                # "s" in the outgoing edge name indicates that it points to a splitting.
                if "s" not in str(e[1]):
                    subjet_edges.append(
                        EdgeFromSubjet(
                            e,
                            int(e[1]),
                            jet.subjets[int(e[1])].constituents.four_vectors().sum().pt,
                            jet.subjets[int(e[1])].part_of_iterative_splitting,
                        )
                    )
                else:
                    n = graph.nodes[e[1]]
                    splitting_edges.append(
                        EdgeFromSplitting(e, f"s{n['splitting_index']}" in iterative_splitting_indices_with_prefix)
                    )

            # We need to match up the subjets with the iterative splittings.
            # To do so, we search over the splitting edges, looking for a subjet edge which match the iterative splitting status
            # of the splitting edge. Since we only expect at most one splitting edge to have a true iterative splitting status (because
            # it looks at the next splitting in the graph), this will select the proper subjet. If both splitting edges have a false iterative
            # splitting status, then we'll assign the first splitting edge to the hardest subjet, and so on.
            subjet_edges_sorted_by_pt = sorted(subjet_edges, key=lambda x: x.subjet_pt, reverse=True)
            for splitting_edge in splitting_edges:
                for i, subjet_edge in enumerate(subjet_edges_sorted_by_pt):
                    # Look for the subjet edge matching iterative splitting
                    # Once we found one, we're done.
                    if splitting_edge.part_of_iterative_splitting == subjet_edge.part_of_iterative_splitting:
                        subjet_edge_to_remove = subjet_edges_sorted_by_pt.pop(i)
                        break

                graph.edges[splitting_edge.edge]["subjet_index"] = graph.edges[subjet_edge_to_remove.edge][
                    "subjet_index"
                ]
                nodes_to_remove.append(subjet_edge_to_remove.edge[1])

    # Actually remove the nodes. We can't do it above because it will invalidate the iterator.
    for node in nodes_to_remove:
        graph.remove_node(node)

    # All iterative splittings are green.
    for name in iterative_splitting_indices_with_prefix:
        n = graph.nodes[name]
        n["color"] = "/greens3/3"

    # Label the nodes and edges.
    # Each node is labeled with the kt of the splitting.
    # Each edge is a subjet.
    # Each node which is just a constituent (ie. it has no nodes leaving it) is made white so that it doesn't show up.
    # This is because it's not a splitting, and we don't want to confuse it as such.
    # This information allows us to confirm that the harder branch was always identified.
    for name in graph.nodes:
        # Retrieve the node since it appears that we cannot directly iterate over the nodes.
        n = graph.nodes[name]

        # First, general node styling
        n["penwidth"] = 2.5
        n["fontsize"] = 22

        # First, remove the constituent labels. If it's just a constituent, we want it to be empty.
        n["label"] = ""

        # Add the kt as the main label of a node.
        # NOTE: Not all nodes will have the splitting index because we also end all subjets with a node, which
        #       is obviously not a splitting. In that case, we hide it by make the color white.
        if "splitting_index" in n:
            # logger.debug(f"splitting_index for {n}: {n['splitting_index']}")
            splitting = jet.splittings[int(n["splitting_index"])]
            n["label"] = f"{splitting.kt:.01f}"
        else:
            n["color"] = "white"

        # Skip the next steps for the origin because it doesn't contain any incoming edges
        if name == "s0":
            continue

        # Now deal with the edges (subjets)
        # in_edges are edges which are incoming into the node.
        edges_labels = graph.in_edges(name)
        # Each node should only have one incoming edge
        if len(edges_labels) != 1:
            raise ValueError(
                "Node {n} has too many edges: {len(edges_labels)}. This shouldn't ever happen, so you'll need to check it out!"
            )
        e = list(edges_labels)[0]
        subjet = jet.subjets[int(graph.edges[e]["subjet_index"])]
        # Increase the line with
        graph.edges[e]["penwidth"] = 2
        # If requested, add the constituents pt label to the subjet.
        if show_subjet_pt:
            graph.edges[e]["fontsize"] = 22
            graph.edges[e]["label"] = f"{subjet.constituents.four_vectors().sum().pt:.01f}"
        # Identify iterative splitting edges via the subjet properties.
        if subjet.part_of_iterative_splitting:
            graph.edges[e]["color"] = "/blues3/3"

    # Highlight individual splitting if requested.
    try:
        n = graph.nodes[f"s{selected_splitting_index}"]
        n["color"] = "/reds3/3"
    except KeyError:
        ...

    # Now we want to draw the graph.
    # We convert to graphviz for plotting because networkx requires too much plotting configuration for labels, etc.
    graphviz_graph = nx.nx_agraph.to_agraph(graph)
    # Show the graph from left to right
    graphviz_graph.graph_attr.update(rankdir="LR")
    # Determine and the layout and draw
    graphviz_graph.layout("dot")
    graphviz_graph.draw(str(path / f"{filename}.pdf"))
    logger.info(f"Saved graph to {path / filename}.pdf")

    return graph
