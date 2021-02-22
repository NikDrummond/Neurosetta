import pandas as pd

def swc_input(path):
    """ Given a file path, generate a rossetaNEURON from an swc file
    """
    swc = pd.read_csv(path,
            header = None,
            index_col = False,
            delim_whitespace = True)
    swc.columns = ["sample_number",
                    "structure_Identifier",
                    "x",
                    "y",
                    "z",
                    "radius",
                    "parent_sample"]
                    
    return swc

# def df_input


# def navis_input


# def graph_input


# def Digraph_input