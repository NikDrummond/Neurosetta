import os

# from Neurosetta.graphs import g_has_property
import graph_tool.all as gt
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

# Main core class


class Stone(object):
    # Constructor
    def __init__(self, name: str) -> None:
        self._type = None
        self.name = name
        self.id = id
        self.units = None


class Tree_graph(Stone):
    """
    Tree graph
    """

    def __init__(self, name: str, graph: gt.Graph, units: str = "nm") -> None:
        super().__init__(name)
        self.graph = graph
        self._type = "Neuron"


class Node_table(Stone):
    """
    SWC like node table
    """

    def __init__(self, name: str, nodes: pd.DataFrame) -> None:
        self.nodes = nodes

        super().__init__(name)


class Neuron_mesh(Stone):
    """
    A mesh
    """

    def __init__(
        self, name: str, vertices: np.ndarray[float], faces: np.ndarray[int]
    ) -> None:
        self.vertices = vertices
        self.faces = faces

        super().__init__(name)


# check properties


def g_has_property(
    N: gt.Graph | Tree_graph, g_property: str, t: str | bool = None
) -> bool:
    """
    Check if a property is within a graph. Will either check for a property generally, or can check specifically graph, vertex, or edge.

    If the g_property argument is None (default), all graph, vertex, and edge properties are checked against. If however the looked for property is a specific type, then the g_property argument can be set to either "v", "e", or "g".

    In this case only the graph ('g'), vertex ('v'), or edge ('e') property is checked.
    """
    if isinstance(N, Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # if t is specifically vertex or edge
    if t is not None:
        # check if vertex property
        if t == "v":
            return ("v", g_property) in g.properties
        elif t == "e":
            return ("e", g_property) in g.properties
        elif t == "g":
            return ("g", g_property) in g.properties
    else:
        return (
            (("v", g_property) in g.properties)
            | (("e", g_property) in g.properties)
            | (("g", g_property) in g.properties)
        )


# read swc


def read_swc(path: str, output: str = "Graph") -> List | Tree_graph | Node_table:
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
        sep="\s+",
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

    # populate them
    vprop_rad.a = df.radius.values[inds]
    vprop_coords.set_2d_array(df[["x", "y", "z"]].values[inds].T)

    # add type to nodes - infer from topology rather than from table
    # types
    infer_node_types(g)

    # add them
    g.vp["radius"] = vprop_rad
    g.vp["coordinates"] = vprop_coords

    return g


def infer_node_types(g: gt.Graph, array: bool = False) -> np.ndarray:
    """

    Parameters
    ----------
    g : gt.Graph
        _description_
    internal : bool, optional
        _description_, by default True
    array : bool, optional
        _description_, by default False

    Returns
    -------
    np.ndarray
        _description_
    """
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

    if array:
        return vprop_type.a


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


## .nr file format read/write
def save(N: Tree_graph, f_path=None):
    """_summary_

    Parameters
    ----------
    N : nr.Tree_graph
        _description_
    f_path : _type_, optional
        _description_, by default None
    """
    # check we have the ID property of our graph
    if ("g", "ID") not in N.graph.properties:
        idgp = N.graph.new_gp("string")
        idgp[N.graph] = N.name
        N.graph.gp["ID"] = idgp

    # If no f_path is given, save where we are
    if f_path is None:
        sp = N.name + ".nr"
    else:
        sp = f_path + N.name + ".nr"

    N.graph.save(sp, fmt="gt")


def load(f_path):
    """"""
    g = gt.load_graph(f_path, fmt="gt")
    name = g.gp["ID"]
    return Tree_graph(name, g)


### Prototyping Neuronal network class


class Network_graph(Stone):
    """
    Neuronal Network
    """

    def __init__(self, name: str, graph: gt.Graph) -> None:
        self.graph = graph
        super().__init__(name)
        self.make()

    def make(self):
        self.id_lookup = dict(
            map(lambda i, j: (i, j), self.graph.vp["ids"].a, self.graph.get_vertices())
        )


def Network_from_table(
    edges: pd.DataFrame | np.ndarray,
    add_weights: bool = True,
    weight_label: str = "weight",
    name: str = None,
):
    """
    Build a neuron network graph from a data frame or edge list.

    The first two columns of the edges input must relate to source -> target edge mapping.

    If add_weights is True, the third column of edges will be used as the weighting for each edge.

    Any other columns will be ignored.
    """
    # unpack the weightes form the df
    if isinstance(edges, pd.DataFrame):
        edges = edges.values

    # if weights is false, keep only first two columns
    if not add_weights:
        edges = edges[:, 0:2]
        g = gt.Graph(edges, hashed=True, hash_type="int")
    else:
        g = gt.Graph(
            edges[:, 0:3], hashed=True, eprops=[(weight_label, "int")], hash_type="int"
        )

    return Network_graph(name=name, graph=g)


def add_vp(g, df, id_col, val_col, name, dtype):
    """
    Find a quicker way to do this...
    """
    vprop = g.new_vp(dtype)
    for v in g.iter_vertices():
        vprop[v] = df.loc[df[id_col] == g.vp["ids"][v], val_col].values[0]
    g.vp[name] = vprop


class Forest_graph(Stone):

    def __init__(self, graph: gt.Graph | List | pd.DataFrame, name=None, **kwargs):
        super().__init__(name)
        # make graph
        if isinstance(graph, gt.Graph):
            self.graph = graph
        elif isinstance(graph, List):
            self.graph = self._make_from_list(graph)
        elif isinstance(graph, pd.DataFrame):
            self.graph = self._make_from_edge_table(graph, **kwargs)
        else:
            raise TypeError("Given type for Graph is not recognised")
        self._type = "Forest"

    def _make_from_list(self, N_all: List):
        """Make a graph from a List of neurons"""
        g = gt.Graph()
        # add vertices
        g.add_vertex(len(N_all))
        # add individual neuron class objects as vertex property
        g.vp["Neurons"] = g.new_vp("object", N_all)
        # add core vertex properties properties
        g.vp["ID"] = g.new_vp(
            "string", [g.vp["Neurons"][i].graph.gp["ID"] for i in g.iter_vertices()]
        )
        # core graph properties (has an ID)
        g.gp["ID"] = g.new_gp("string", "unnamed")

        return g

    def _make_from_edge_table(self, edges, add_weights=True, add_types=True):

        # make graph
        if not add_weights:
            edges = edges[["pre", "post"]].values
            g = gt.Graph(edges, hashed=True, hash_type="int")
        else:
            g = gt.Graph(
                edges[["pre", "post", "weight"]].values,
                hashed=True,
                eprops=[("weight", "int")],
                hash_type="int",
            )

        # if we want to add types try to add them
        if add_types:
            if "Input_type" in edges.columns and "Output_type" in edges.columns:
                type_vp = g.new_vp("string")

                for v in g.iter_vertices():
                    type_vp[v] = _find_n_type(g.vp["ids"][v], edges)

                g.vp["type"] = type_vp
            else:
                raise AttributeError("Input_type and Output_type not in edges column")

        return g

    ### Propagate properties from a gp key or list of keys
    def propagate_vp_to_gp(self, props):
        if isinstance(props, str):
            if g_has_property(self.graph.vp["Neurons"][0], props):
                curr_type = _get_property_type(
                    self.graph.vp["Neurons"][0].graph.gp[props]
                )
                self.graph.vp[props] = self.graph.new_vp(
                    curr_type,
                    [
                        self.graph.vp["Neurons"][i].graph.gp[props]
                        for i in self.graph.iter_vertices()
                    ],
                )
            else:
                print("First neuron does not have " + props + " property, so skipping")
        elif isinstance(props, List):
            for current_prop in props:
                if g_has_property(self.graph.vp["Neurons"][0], current_prop):
                    curr_type = _get_property_type(
                        self.graph.vp["Neurons"][0].graph.gp[current_prop]
                    )
                    self.graph.vp[current_prop] = self.graph.new_vp(
                        curr_type,
                        [
                            self.graph.vp["Neurons"][i].graph.gp[current_prop]
                            for i in self.graph.iter_vertices()
                        ],
                    )
                else:
                    print(
                        "First neuron does not have "
                        + current_prop
                        + " property, so skipping"
                    )

    def add_type_lookup(self):
        self.types = np.array([self.graph.vp['type'][i] for i in self.graph.iter_vertices()],dtype = str)

    def add_id_lookup(self):
        self.ids = np.array([self.graph.vp['ids'][i] for i in self.graph.iter_vertices()],dtype = int)
        
    def type_subset_ind(self,n_type):
        if not hasattr(self,'types'):
            self.add_type_lookup()
        return np.where(self.types == n_type)[0]

    def node_synapse_by_type(self, v, direction: str = "both"):
        if direction == "out":
            edges = self.graph.get_out_edges(v)
        elif direction == "in":
            edges = self.graph.get_in_edges(v)
        elif direction == "both":
            edges = self.graph.get_all_edges(v)
        else:
            raise AttributeError('Direction must be Im, Out, or Both')

        # in types and ids
        in_ids = [self.graph.vp["ids"][vertex] for vertex in edges[:, 0]]
        in_types = [self.graph.vp["type"][vertex] for vertex in edges[:, 0]]
        # out types and ids
        out_ids = [self.graph.vp["ids"][vertex] for vertex in edges[:, 1]]
        out_types = [self.graph.vp["type"][vertex] for vertex in edges[:, 1]]
        # synapse counts
        syn_counts = [self.graph.ep["weight"][e] for e in edges]
        # df
        df = pd.DataFrame(
            {
                "Input_id": in_ids,
                "Input_type": in_types,
                "Output_id": out_ids,
                "Output_types": out_types,
                "Synapses": syn_counts,
            }
        )
        # return
        return df

    def synapses_by_type(self, subset: None | List = None, direction: str = 'out'):

        # initialise output list
        # outputs_dfs = []
        # if subset is none use all neurons
        if subset is None:
            subset = self.graph.get_vertices()

        df = pd.concat([self.node_synapse_by_type(v,direction = direction) for v in tqdm(subset)])

        return df

    def all_coords(self):
        """ get the point cloud of all points"""
        coords = 

def _get_property_type(prop):
    """dumb function to get string for property maps given some property"""
    if isinstance(prop, str):
        return "string"
    elif isinstance(prop, int):
        return "int"
    elif isinstance(prop, float):
        return "double"


def _find_n_type(n_id, edges):
    try:
        t = edges.loc[edges.pre == n_id, "Input_type"].values[0]
    except:
        t = edges.loc[edges.post == n_id, "Output_type"].values[0]
    return t
