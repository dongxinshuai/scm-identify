import pandas as pd
import json
import argparse
from utils.logger import LOGGER
from scenarios import dgm_by_scenario
from utils.GraphDrawer import printGraph
import numpy as np
from ParameterLearning.linear_parameter_identification import get_F_bin_and_varnames_from_dotgraph, parameter_by_trek, parameter_standard, add_edge_coefficients_to_graph
import os
import pydot

if __name__ == "__main__":    

    parser = argparse.ArgumentParser("params")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--method", type=str, default='trek')
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--s", type=str, default="multitasking") 
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dot_path", type=str)
    args = parser.parse_args()

    print(f"Searching Scenario {args.s}...")
    dgm = dgm_by_scenario[args.s]
    df_x, _ = dgm.generate_data(N=args.n, normalized=True)

    dotgraph = pydot.graph_from_dot_file(args.dot_path)[0]
    F_bin, var_name_ls = get_F_bin_and_varnames_from_dotgraph(dotgraph)

    if args.method == 'trek':
        F, nll = parameter_by_trek(df_x, F_bin, var_name_ls, device=args.device, lr=args.lr, epochs=args.epochs)
    elif args.method == 'standard':
        F, nll = parameter_standard(df_x, F_bin, var_name_ls, device=args.device, lr=args.lr, epochs=args.epochs)

    add_edge_coefficients_to_graph(dotgraph, F_bin, F, var_name_ls)

    new_file_name = f"{args.dot_path.split('.dot')[0]}_param_by_{args.method}_nll{nll}"

    dotgraph.write_raw(new_file_name+'.dot')
    dotgraph.set_size(f'"{100},{100}!"')
    dotgraph.set_layout(f'dot')
    dotgraph.write_png(new_file_name+'.png')
