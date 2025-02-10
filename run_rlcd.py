import pandas as pd
import json
import argparse
from utils.logger import LOGGER
from scenarios import dgm_by_scenario
from utils.GraphDrawer import printGraph
import numpy as np
from StructureLearning.RLCD.RLCD_alg import RLCD
from utils.OraclePartialCorrTest import OraclePartialCorrTest
from utils.OracleRankTest import OracleRankTest
from utils.Chi2RankTest import Chi2RankTest
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser("main")
    parser.add_argument("--s", type=str, default="multitasking")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--stage1_ges_sparsity", type=float, default=2)
    parser.add_argument("--stage1_partition_thres", type=float, default=3)
    parser.add_argument("--stage1_method", type=str, default="fges")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--rank_test_N_scaling", type=float, default=1)
    
    args = parser.parse_args()

    dgm = dgm_by_scenario[args.s]
    
    if args.sample:
        df_x, df_v = dgm.generate_data(N=args.n, normalized=True)
        input_parameters = {
            "ranktest_method": Chi2RankTest(df_x.to_numpy(), args.rank_test_N_scaling),
            "citest_method": None,
            "stage1_method": args.stage1_method,
            "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
            "stage1_ges_sparsity": args.stage1_ges_sparsity,
            "stage1_partition_thres": args.stage1_partition_thres
        }
        result_dotgraph, result_stage1_dotgraph, _, _ = RLCD(args.sample, dgm.xvars, df_x, input_parameters)
    else:
        input_parameters = {
            "ranktest_method": OracleRankTest(dgm),
            "citest_method": OraclePartialCorrTest(dgm, np.zeros((1, len(dgm.xvars)))),
            "stage1_method": args.stage1_method,
            "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
            "stage1_ges_sparsity": args.stage1_ges_sparsity,
            "stage1_partition_thres": args.stage1_partition_thres
        }
        result_dotgraph, result_stage1_dotgraph, _, _ = RLCD(args.sample, dgm.xvars, None, input_parameters)

    plots_save_path = f'{args.s}_results'
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)

    if args.sample: 
        printGraph(result_dotgraph, f'{plots_save_path}/alpha{args.alpha}_rtscale{args.rank_test_N_scaling}_N{args.n}.png')


