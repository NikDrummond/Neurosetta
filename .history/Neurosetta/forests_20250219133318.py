import pandas as pd
from typing import List
from numpy import array, where

from Neurosetta import Forest_graph, g_has_property


def _property_error(n: int, prop: str, ids: List):
    # Format only the first n ids in a readable way:
    formatted_ids = ", ".join(map(str, ids[:n]))

    error_message = (
        f"{n} neurons do not have {prop}, Use N_missing_property for a full list of ids. \n" 
        f"printing first {n} ids: [{formatted_ids}]"
    )

    raise ValueError(error_message)


def _all_property_error(prop):
    raise ValueError(f"No neurons in given Forest have {prop} property")

def N_missing_properties(N_all: Forest_graph, prop: str) -> List:
    bool_props = array(
        [
            g_has_property(N_all.graph.vp["Neurons"][v], prop)
            for v in N_all.graph.iter_vertices()
        ],
        dtype=bool,
        )
    ids = [int(N_all.graph.vp['Neurons'][v].name) for v in where(bool_props == False)[0]]
    return ids

def check_neurons_have_property(
    N_all: Forest_graph, prop: str, tol: int = 0, verbose: bool = True
) -> bool:
    bool_props = array(
        [
            g_has_property(N_all.graph.vp["Neurons"][v], prop)
            for v in N_all.graph.iter_vertices()
        ],
        dtype=bool,
    )
    # count fails
    fails = abs( sum(bool_props) - len(bool_props))
    
    if fails <= tol:
        return True
    else:
        if verbose:
            if fails == len(bool_props):
                # _all_property_error(prop)
                ids = [int(N_all.graph.vp['Neurons'][v].name) for v in where(bool_props == False)[0]]
                _property_error(tol, prop, ids)
            else:
                # get ids which fail
                ids = [int(N_all.graph.vp['Neurons'][v].name) for v in where(bool_props == False)[0]]
                _property_error(tol, prop, ids)
        else:
            return False
        

def _get_neuron_input_table(N_all: Forest_graph, include_partner_type: bool = True, return_missing: bool = False, tol: int = 0):
    

    # get missing ids - we will return this along side the inputs if we pass tolerence
    missing_ids = N_missing_properties(N_all, 'inputs')
    # if all are missing raise an error
    if len(missing_ids) == N_all.graph.num_vertices():
        _all_property_error('inputs')
    # if more than tol are missing, also raise error
    elif len(missing_ids) > tol:
        _property_error(len(missing_ids, 'inputs', missing_ids))

    # now that is out of the way, if we have made it this far, we can make our table
    dfs = []
    for v in N_all.graph.iter_vertices():
        try:
            N = N_all.graph.vp['Neurons'][v]
            tmp_df = N.graph.gp['inputs']
            tmp_df['post'] = int(N.name)
            if include_partner_type:
                tmp_df['Partner_type'] = tmp_df['Input_type']
                dfs.append(tmp_df[['pre','post','graph_x','graph_y','graph_z','Partner_type']])
            else:
                dfs.append(tmp_df[['pre','post','graph_x','graph_y','graph_z']])
        except:    
            pass
    # concat data frame
    df = pd.concat(dfs)

    if return_missing:
        # if nothing is missing, say so and just return df
        if len(missing_ids) == 0:
            print('No neurons missing inputs, just returning table')
            return df
        else:
            n = len(missing_ids)
            print(f'{n} neurons are missing input tables, returning list of ids with table')
            return df, missing_ids
    else:
        return df

def _get_neuron_output_table(N_all: Forest_graph, include_partner_type: bool = True, return_missing: bool = False, tol: int = 0):
    

    # get missing ids - we will return this along side the inputs if we pass tolerence
    missing_ids = N_missing_properties(N_all, 'outputs')
    # if all are missing raise an error
    if len(missing_ids) == N_all.graph.num_vertices():
        _all_property_error('outputs')
    # if more than tol are missing, also raise error
    elif len(missing_ids) > tol:
        _property_error(len(missing_ids, 'outputs', missing_ids))

    # now that is out of the way, if we have made it this far, we can make our table
    dfs = []
    for v in N_all.graph.iter_vertices():
        try:
            N = N_all.graph.vp['Neurons'][v]
            tmp_df = N.graph.gp['outputs']
            tmp_df['pre'] = int(N.name)
            dfs.append(tmp_df[['pre','post','graph_x','graph_y','graph_z']])
            if include_partner_type:
                tmp_df['Partner_type'] = tmp_df['Output_type']
                dfs.append(tmp_df[['pre','post','graph_x','graph_y','graph_z','Partner_type']])
            else:
                dfs.append(tmp_df[['pre','post','graph_x','graph_y','graph_z']])
        except:    
            pass
    # concat data frame
    df = pd.concat(dfs)

    if return_missing:
        # if nothing is missing, say so and just return df
        if len(missing_ids) == 0:
            print('No neurons missing outputs, just returning table')
            return df
        else:
            n = len(missing_ids)
            print(f'{n} neurons are missing outputs tables, returning list of ids with table')
            return df, missing_ids
    else:
        return df

def Neuron_synapse_table(N_all: Forest_graph, direction:str = 'all', include_partner_type:bool = True,return_missing:bool = False):
"""
Generate a synapse table for neurons in a Forest_graph based on the specified direction.

Parameters:
    N_all (Forest_graph): A Forest_graph instance containing neuron synaptic data.
    direction (str): The direction of synapse data to include. Accepted values are 'inputs', 'outputs', or 'all'.
    include_partner_type (bool): If True, include the partner type column in the resulting table.
    return_missing (bool): If True, also return lists of neuron IDs that lack the specified synapse properties.

Returns:
    pd.DataFrame or tuple:
        - When return_missing is False, returns a DataFrame concatenating the synapse data.
        - When direction is 'all' and return_missing is True, returns a tuple containing the combined table and lists of missing input and output neuron IDs.

Raises:
    ValueError: If an invalid direction is provided.
"""
    valid_directions = ['all','inputs','outputs']
    if direction == 'inputs':
        return _get_neuron_input_table(N_all, include_partner_type=include_partner_type,return_missing=return_missing)
    elif direction == 'outputs':
        return _get_neuron_output_table(N_all, include_partner_type=include_partner_type,return_missing=return_missing)
    elif direction == 'all':
        in_df = _get_neuron_input_table(N_all, include_partner_type=include_partner_type,return_missing=False)
        out_df = _get_neuron_output_table(N_all, include_partner_type=include_partner_type,return_missing=False)
        df = pd.concat([in_df, out_df])

        if return_missing:
            in_missing = N_missing_properties(N_all,'inputs')
            out_missing = N_missing_properties(N_all,'outputs')
            return df, in_missing, out_missing
        else:
            return df
    else:
        raise ValueError(f"Invalid direction '{direction}'. Expected one of {valid_directions}.")

