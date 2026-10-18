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
    F_bin, var_name_ls, var_name_ls_new = get_F_bin_and_varnames_from_dotgraph(dotgraph) #var_name_ls_new with out "", e.g., in var_name_ls var name may be "L1" but in var_name_ls_new it is L1

    # in df_x we use var name without "" so use var_name_ls_new

    if args.method == 'trek':
        F, nll = parameter_by_trek(df_x, F_bin, var_name_ls_new, device=args.device, lr=args.lr, epochs=args.epochs)
    elif args.method == 'standard':
        F, nll = parameter_standard(df_x, F_bin, var_name_ls_new, device=args.device, lr=args.lr, epochs=args.epochs)

    # when added back to graph we still use var_name_ls, note that in var_name_ls it may originally be without ""
    
    add_edge_coefficients_to_graph(dotgraph, F_bin, F, var_name_ls)

    new_file_name = f"{args.dot_path.split('.dot')[0]}_param_by_{args.method}_nll{nll}"

    dotgraph.write_raw(new_file_name+'.dot')
    dotgraph.set_size(f'"{100},{100}!"')
    dotgraph.set_layout(f'dot')
    dotgraph.write_png(new_file_name+'.png')

    edge_coefficients_df = pd.DataFrame(F, index=var_name_ls, columns=var_name_ls)
    edge_coefficients_df = edge_coefficients_df.round(3)
    excel_file_name = f'{new_file_name}.xlsx'
    sheet_name = 'Edge Coefficients (row->col)'
    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
    edge_coefficients_df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=True)


    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    (num_data_rows, num_data_cols) = edge_coefficients_df.shape
    max_row_excel = num_data_rows
    max_col_excel = num_data_cols # We account for the index later in column_settings

    # Prepare columns for the Excel Table.
    # The first column will be the DataFrame index.
    column_settings = [{'header': edge_coefficients_df.index.name if edge_coefficients_df.index.name else 'Variable'}]
    column_settings.extend([{'header': col} for col in edge_coefficients_df.columns])

    # Add the Excel table structure.
    # The table range starts at A1 (0, 0) and extends to cover all data plus the header row
    worksheet.add_table(0, 0, max_row_excel, max_col_excel, {
        'columns': column_settings,
        'name': 'CorrelationTable',
        'style': 'TableStyleMedium14', # A nice blue style
        'first_column': True # Highlight the first column (your variable names)
    })

    # Optional: Adjust column widths for better readability
    # Iterate through all columns, including the index column
    for i, col_name in enumerate(edge_coefficients_df.columns):
        # Width for data columns
        max_len = max(edge_coefficients_df[col_name].astype(str).map(len).max(), len(col_name))
        worksheet.set_column(i + 1, i + 1, max_len + 2) # +1 because of the index column

    # Adjust width for the index column (first column in Excel)
    if edge_coefficients_df.index.name:
        index_col_header_len = len(edge_coefficients_df.index.name)
    else:
        index_col_header_len = len('Variable') # Default name if none set
    max_index_len = max(edge_coefficients_df.index.astype(str).map(len).max(), index_col_header_len)
    worksheet.set_column(0, 0, max_index_len + 2)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
