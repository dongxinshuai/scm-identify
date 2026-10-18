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
from utils.Chi2ConditionalRankTest import Chi2ConditionalRankTest
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser("main")
    parser.add_argument("--s", type=str, default="BioData")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--stage1_ges_sparsity", type=float, default=2)
    parser.add_argument("--stage1_partition_thres", type=float, default=3)
    parser.add_argument("--stage1_method", type=str, default="fges")
    parser.add_argument("--stage1_dot_path", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--rank_test_N_scaling", type=float, default=1)
    parser.add_argument("--multi_processing", type=int, default=0)
    parser.add_argument("--allow_nonleafx", type=int, default=1)
    parser.add_argument("--maxk", type=int, default=3)
    parser.add_argument("--use_lcdnod", type=int, default=0)
    args = parser.parse_args()

    dgm = dgm_by_scenario[args.s]
    
    if args.sample:
        
        if not args.use_lcdnod:

            df_x, df_v = dgm.generate_data(N=args.n, normalized=True)
            input_parameters = {
                "ranktest_method": Chi2RankTest(df_x.to_numpy(), args.rank_test_N_scaling),
                "citest_method": None,
                "stage1_method": args.stage1_method,
                "stage1_dot_path": args.stage1_dot_path,
                "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
                "stage1_ges_sparsity": args.stage1_ges_sparsity,
                "stage1_partition_thres": args.stage1_partition_thres,
                "multi_processing": args.multi_processing,
                "maxk": args.maxk,
                "allow_nonleafx": args.allow_nonleafx,
            }

            result_dotgraph, result_stage1_dotgraph, rlcd_result_Adj, rlcd_result_vars = RLCD(args.sample, dgm.xvars, df_x, input_parameters)
            
            #from utils.eval_structure import eval_skeleton
            #gtxvars_idxs_in_dgmgt = [dgm.vars.index(t) for t in dgm.xvars]
            #gtxvars_idxs_in_rlcd_result = [rlcd_result_vars.index(t) for t in dgm.xvars]
            #recall, precision, f1, shd = eval_skeleton(dgm.get_causallearn_adj(), gtxvars_idxs_in_dgmgt, rlcd_result_Adj, gtxvars_idxs_in_rlcd_result)
            #print(f"RLCD done, f1 {f1} shd {shd}")
        
        else: 

            df_x_list, df_v_list = dgm.generate_data(N=args.n, normalized=True)

            combined_df_x = pd.concat(df_x_list, axis=0, ignore_index=True)

            input_parameters = {
                "ranktest_method": Chi2ConditionalRankTest([df_x.to_numpy() for df_x in df_x_list], args.rank_test_N_scaling),
                "citest_method": None,
                "stage1_method": args.stage1_method,
                "stage1_dot_path": args.stage1_dot_path,
                "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
                "stage1_ges_sparsity": args.stage1_ges_sparsity,
                "stage1_partition_thres": args.stage1_partition_thres,
                "multi_processing": args.multi_processing,
                "maxk": args.maxk,
                "allow_nonleafx": args.allow_nonleafx,
            }

            result_dotgraph, result_stage1_dotgraph, rlcd_result_Adj, rlcd_result_vars = RLCD(args.sample, dgm.xvars, combined_df_x, input_parameters)



    else:
        input_parameters = {
            "ranktest_method": OracleRankTest(dgm),
            "citest_method": OraclePartialCorrTest(dgm, np.zeros((1, len(dgm.xvars)))),
            "stage1_method": args.stage1_method,
            "stage1_dot_path": args.stage1_dot_path,
            "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
            "stage1_ges_sparsity": args.stage1_ges_sparsity,
            "stage1_partition_thres": args.stage1_partition_thres,
            "multi_processing": args.multi_processing,
            "maxk": args.maxk,
            "allow_nonleafx": args.allow_nonleafx,
        }
        result_dotgraph, result_stage1_dotgraph, _, _ = RLCD(args.sample, dgm.xvars, None, input_parameters)

    plots_save_path = f'results_{args.s}'
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)

    if args.sample: 
        printGraph(result_stage1_dotgraph, f'{plots_save_path}/result_stage1{args.stage1_method}_sparsity{args.stage1_ges_sparsity}.png')
        printGraph(result_dotgraph, f'{plots_save_path}/result_stage1{args.stage1_method}_sparsity{args.stage1_ges_sparsity}_alpha{args.alpha}_ranktestscale{args.rank_test_N_scaling}_allow_nonleafx{args.allow_nonleafx}_maxk{args.maxk}.png')


