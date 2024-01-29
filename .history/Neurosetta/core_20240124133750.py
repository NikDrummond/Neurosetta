import os
import graph_tool.all as gt
import numpy as np
import pandas as pd
from typing import List, Optional

class Stone:
    """Base class for Stone objects."""

    def __init__(self, name: str) -> None:
        """
        Initialize a Stone object with the given name.

        Args:
            name (str): The name of the Stone object.
        """
        self._name = name
        self._id = None

    @property
    def name(self) -> str:
        """Return the name of the Stone object."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the Stone object."""
        if not isinstance(value, str):
            raise TypeError("Name must be a string.")
        self._name = value

    @property
    def id(self) -> Optional[int]:
        """Return the id of the Stone object."""
        return self._id

    @id.setter
    def id(self, value: Optional[int]) -> None:
        """Set the id of the Stone object."""
        if value is not None and not isinstance(value, int):
            raise TypeError("ID must be an integer or None.")
        self._id = value

    def __repr__(self) -> str:
        """Return a string representation of the Stone object."""
        return f"<{self.__class__.__name__} name={self._name}, id={self._id}>"


class TreeGraph(Stone):
    """Tree graph class based on graph-tool library."""

    def __init__(self, name: str, graph: gt.Graph) -> None:
        """
        Initialize a TreeGraph object with the given name and graph.

        Args:
            name (str): The name of the TreeGraph object.
            graph (gt.Graph): The graph-tool Graph object.
        """
        self.graph = graph

        super().__init__(name)

        self._check_graph(graph)

    def _check_graph(self, graph: gt.Graph) -> None:
        """Check if the input graph is directed and simple."""
        if not graph.is_directed():
            raise ValueError("Graph must be directed.")
        if not graph.is_simple():
            raise ValueError("Graph must be simple.")

    def __repr__(self) -> str:
        """Return a string representation of the TreeGraph object."""
        return f"<{self.__class__.__name__} name={self._name}, id={self._id}, num_vertices={self.graph.num_vertices()}, num_edges={self.graph.num_edges()}>"


class NodeTable(Stone):
    """SWC-like node table class."""

    def __init__(self, name: str, nodes: pd.DataFrame) -> None:
        """
        Initialize a NodeTable object with the given name and nodes DataFrame.

        Args:
            name (str): The name of the NodeTable object.
            nodes (pd.DataFrame): The pandas DataFrame containing the node table.
        """
        self.nodes = nodes

        super().__init__(name)

        self._check_nodes(nodes)

    def _check_nodes(self, nodes: pd.DataFrame) -> None:
        """Check if the input nodes DataFrame is not empty."""
        if nodes.empty:
            raise ValueError("Node table cannot be empty.")

    def __repr__(self) -> str:
        """Return a string representation of the NodeTable object."""
        return f"<{self.__class__.__name__} name={self._name}, id={self._id}, num_rows={self.nodes.shape[0]}>"


class NeuronMesh(Stone):
    """
    A 3D mesh for a neuron.
    """

    def __init__(self name: str, vertices:.ndarray, faces: np.ndarray):
        self.name = name
        self.vertices = vertices
        self.faces = faces

        # Ensure that the vertices and faces arrays are numpy arrays
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

        # Call the constructor of the superclass if it exists
        try:
            super().__init__(name)
        except TypeError:
            pass


# read swc


def read_swc(path: str, output: str = "Table") -> List | Tree_graph | Node_table:
    """
    Read in swc file(s) and return table/graph output
    """

    if os.path.isdir(path):
        files = os.listdir(path)
        files = [f for f in files if f.endswith(".swc")]
        N = [_read_swc(path + f, output=output) for f in files]

    else:
        N = _read_swc(path, output=output)

    return N


def _read_swc(file_path: str, output: str = "Table") -> Tree_graph | Node_table:
    name = os.path.splitext(os.path.basename(file_path))[0]
    df = table_from_swc(file_path)

    if output.casefold() == "Table".casefold():
        return Node_table(name=name, nodes=df)

    elif output.casefold() == "Graph".casefold():
        g = graph_from_table(df)
        return Tree_graph(name=name, graph=g)


# table from swc
def table_from_swc(file_path: str) -> pd.DataFrame:
    """
    Node table from swc file, using pandas
    """
    df = pd.read_csv(
        file_path,
        names=["node_id", "type", "x", "y", "z", "radius", "parent_id"],
        comment="#",
        engine="c",
        delim_whitespace=True,
        dtype={
            "node_id": np.int32,
            "type": np.int32,
            "x": np.float64,
            "y": np.float64,
            "z": np.float64,
            "radius": np.float64,
            "parent_id": np.int32,
        },
        index_col=False,
    )

    # check for duplicate node ids
    if not np.unique(df.node_id.values).size == len(df.node_id.values):
        raise AttributeError("Duplicate Node Ids found")

    return df


# Graph from table
def _node_inds(g: gt.Graph, df: pd.DataFrame) -> List[int]:
    """
    Given a graph, with ids as a vp, and a df with the same set of ides, find the order of indicies to order things going from the table to the graph

    This is important whenever adding an attribute to nodes in a graph
    """
    # node id orders - this is the order of nodes in the graph, which match the node ids in the table
    ids = g.vp["ids"].a
    # nodes in the table
    nodes = df["node_id"].values

    inds = [np.where(nodes == i)[0][0] for i in ids]

    return inds


def graph_from_table(df: pd.DataFrame) -> gt.Graph:
    """
    From a node table, generate a graph-tool graph
    """

    if isinstance(df, Node_table):
        df = df.nodes.copy()
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise AttributeError("Input type not recognised")
    # get edges (without root)
    edges = df.loc[df.parent_id != -1, ["parent_id", "node_id"]].values

    # create new (hashed) graph with edges
    g = gt.Graph(edges, hashed=True, hash_type="int")

    # we want to know the indicies of the nodes in the table that match how the nodes were added to the graph

    # indicies of graph nodes in table
    inds = _node_inds(g, df)

    # add some attributed from node table
    # initilise vertex properties - radius, coordinates
    vprop_rad = g.new_vp("double")
    vprop_coords = g.new_vp("vector<double>")

    # populate them - i think the error is here??
    vprop_rad.a = df.radius.values[inds]
    vprop_coords.set_2d_array(df[["x", "y", "z"]].values[inds].T)

    # add type to nodes - infer from topology rather than from table
    # types
    out_deg = g.get_out_degrees(g.get_vertices())
    in_deg = g.get_in_degrees(g.get_vertices())
    ends = np.where(out_deg == 0)
    branches = np.where(out_deg > 1)
    root = np.where(in_deg == 0)
    node_types = np.zeros_like(g.get_vertices())
    node_types[ends] = 6
    node_types[branches] = 5
    node_types[root] = -1

    # create and add populate property
    vprop_type = g.new_vp("int")
    vprop_type.a = node_types
    g.vp["type"] = vprop_type

    # add them
    g.vp["radius"] = vprop_rad
    g.vp["coordinates"] = vprop_coords

    return g


# graph to node table
def graph_to_table(
    g: gt.Graph, output: str = "Neurosetta"
) -> pd.DataFrame | Node_table:
    """
    Convert Tree Graph to swc like table
    """

    # if Tree Graph
    if isinstance(g, Tree_graph):
        name = g.name
        g = g.graph
    elif isinstance(g, gt.Graph):
        name = None
    else:
        raise AttributeError("input type not supported")

    # generate a node table from a graph.
    # node ids
    ids = g.vp["ids"].a
    radius = g.vp["radius"].a
    # is there a better way to do this??
    coords = np.asarray([g.vp["coordinates"][i] for i in g.iter_vertices()])
    # generate parent node ids
    parents = np.zeros_like(ids)
    for i in range(len(ids)):
        v = g.get_vertices()[i]
        # get parent
        p = g.get_in_neighbors(v)
        if len(p) != 1:
            p_id = -1
        else:
            p_id = g.vp["ids"][p[0]]
        parents[i] = p_id

    Type = g.vp["type"].a

    df = (
        pd.DataFrame(
            {
                "node_id": ids,
                "type": Type,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "radius": radius,
                "parent_id": parents,
            }
        )
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    if output.casefold() == "Neurosetta".casefold():
        return Node_table(name=name, nodes=df)
    elif output.casefold() == "Table".casefold():
        return df


def write_swc(N: Node_table | Tree_graph, fpath: str) -> None:
    """
    Write Neuron to swc
    """
    if isinstance(N, Tree_graph):
        N = graph_to_table(N)

    np.savetxt(
        fpath,
        N.nodes,
        header="SWC Generated using Neurosetta \n Columns \n" + str(N.nodes.columns),
    )
