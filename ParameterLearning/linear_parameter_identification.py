import numpy as np
import torch
import torch.nn as nn
import pydot
import argparse
import time


def get_cov_V_by_trek(F_bin, F, diag_population_cov_V, records=None):

    def get_directed_pathes(fa, middle_res, ch, res_ls):
        node_num = F_bin.shape[0]

        if fa==ch:
            res_ls.append(middle_res)
            return

        for direct_ch in range(node_num):
            if F_bin[fa, direct_ch]==1:
                temp_middle_res = middle_res.copy()
                temp_middle_res[fa, direct_ch]=1
                get_directed_pathes(direct_ch, temp_middle_res, ch, res_ls)

        return

    def check_intersection(directed_path_to_i, directed_path_to_j, common_fa):
        node_num = F_bin.shape[0]

        set_path_to_i = set()
        set_path_to_i.add(common_fa)
        set_path_to_j = set()
        set_path_to_j.add(common_fa)

        for ii in range(node_num):
            for jj in range(node_num):
                if directed_path_to_i[jj,ii]:
                    set_path_to_i.add(ii)
                    set_path_to_i.add(jj)

                if directed_path_to_j[ii,jj]:
                    set_path_to_j.add(ii)
                    set_path_to_j.add(jj)

        intersection = set_path_to_i.intersection(set_path_to_j)
        intersection.remove(common_fa)

        return True if len(intersection)==0 else False


    def get_cov(node_i, node_j, record=None):

        node_num = F_bin.shape[0]
        cov = 0

        if record is None:
            record = []
            for common_fa in range(node_num):
                directed_pathes_to_j_ls = []
                get_directed_pathes(common_fa, np.zeros_like(F_bin).astype(bool), node_j, directed_pathes_to_j_ls)
                directed_pathes_to_i_ls = []
                get_directed_pathes(common_fa, np.zeros_like(F_bin).astype(bool), node_i, directed_pathes_to_i_ls)

                for directed_path_to_j in directed_pathes_to_j_ls:
                    for directed_path_to_i in directed_pathes_to_i_ls:
                        if check_intersection(directed_path_to_i, directed_path_to_j, common_fa):
                            cov += diag_population_cov_V[common_fa] * F[torch.from_numpy(directed_path_to_i).to(F.device)].prod()\
                                  * F[torch.from_numpy(directed_path_to_j).to(F.device)].prod()
                            record.append((common_fa,directed_path_to_i,directed_path_to_j))
        else:
            for common_fa, directed_path_to_i, directed_path_to_j in record:
                cov += diag_population_cov_V[common_fa] * F[torch.from_numpy(directed_path_to_i).to(F.device)].prod()\
                      * F[torch.from_numpy(directed_path_to_j).to(F.device)].prod()
        return cov, record
    #############################################################   
         
    node_num = F_bin.shape[0]
    cov_V=torch.zeros(node_num,node_num)
    for i in range(node_num):
        cov_V[i,i]=diag_population_cov_V[i]

    if records is None:
        records = [[None for i in range(node_num)] for j in range(node_num)]

        for i in range(node_num):
            for j in range(node_num):
                if j==i:
                    continue
                elif j>i:
                    cov, record_i_j = get_cov(i, j)
                    records[i][j] = record_i_j
                    cov_V[i,j] = cov
                    cov_V[j,i] = cov

    else:
        for i in range(node_num):
            for j in range(node_num):
                if j==i:
                    continue
                elif j>i:
                    
                    record_i_j = records[i][j]
                    cov, record_i_j = get_cov(i, j, record=record_i_j)
                    cov_V[i,j] = cov
                    cov_V[j,i] = cov

    return cov_V, records

def parameter_standard(df, F_bin, var_name_ls, device='cpu', lr=0.02, epochs=100):

    F_param = torch.empty(F_bin.shape).to(device)
    nn.init.kaiming_normal_(F_param)
    F_param = torch.tanh(F_param)
    F_param.requires_grad_()
    Omega_param = nn.Parameter(torch.ones(len(var_name_ls)))
    Omega_param.requires_grad_()

    params = [F_param, Omega_param] 
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-6)
    emp_cov_X = torch.from_numpy(df.cov().to_numpy())


    x_indexes = []
    for x_name in df.columns:
        index = var_name_ls.index(x_name)
        x_indexes.append(index)


    for i in range(epochs):
        t1 = time.time()
        optimizer.zero_grad()

        with torch.enable_grad():

            mask = Omega_param.clone().detach()
            mask[x_indexes] = 1
            Omega = Omega_param / mask
            F = F_param*torch.tensor(F_bin)
            I = torch.eye(len(var_name_ls))
            Q = torch.linalg.inv(I - F.T)
            cov_V = Q @ torch.diag(Omega.to(Q.dtype)) @ Q.T
            cov_X = ((cov_V[x_indexes].T)[x_indexes]).T

            loglikelihood = -0.5*torch.slogdet(cov_X)[1]-0.5*torch.trace(torch.mm(torch.inverse(cov_X).to(emp_cov_X.dtype), emp_cov_X))
            nll= -loglikelihood 
            #penalty = torch.max(torch.zeros_like(F),torch.abs(F*torch.tensor(F_bin))-0.99*torch.ones_like(F)).pow(2).sum()
            #+ torch.min(torch.zeros_like(F),torch.abs(F)-0.005*torch.ones_like(F)).pow(2).sum()
            loss = nll # + penalty*100


        loss.backward()
        optimizer.step()
        
        print(f"iter {i} loss {loss}, nll {nll}")
        print("mse", ((cov_X-emp_cov_X).pow(2)).mean())
        print(f"time {time.time()-t1}")

    print("End")
    print("cov_X", cov_X)
    print("emp_cov_X", emp_cov_X)

    F = (F.detach().numpy())*F_bin
    mask = Omega_param.clone().detach()
    mask[x_indexes] = 1
    Omega = Omega_param / mask
    Omega = Omega.detach().numpy()
    
    import sys
    sys.path.insert(0, '..')
    from DGM.LinearSCM import LinearSCM
    dgm = LinearSCM()
    dgm.vars = var_name_ls
    xvars = []
    lvars = []
    for var_name in var_name_ls:
        if var_name in df.columns:
            xvars.append(var_name)
        else:
            lvars.append(var_name)
    dgm.xvars = xvars
    dgm.lvars = lvars
    dgm.F = F 
    # variance of error terms
    dgm.omega = [x for x in Omega] 
    dgm.normalize_to_have_unit_variance(normalize_var_type='x_and_l')

    return dgm.F, nll.detach().numpy()



def parameter_by_trek(df, F_bin, var_name_ls, device='cpu', lr=0.02, epochs=100):

    F = torch.empty(F_bin.shape).to(device)
    nn.init.kaiming_normal_(F)
    F = torch.tanh(F)
    F.requires_grad_()

    params = [F] 
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-6)
    emp_cov_X = torch.from_numpy(df.cov().to_numpy())
    diag_population_cov_V = torch.ones(len(F_bin))

    x_indexes = []
    for x_name in df.columns:
        index = var_name_ls.index(x_name)
        diag_population_cov_V[index] = df.cov()[x_name][x_name]
        x_indexes.append(index)
    
    records = None

    for i in range(epochs):
        t1 = time.time()
        optimizer.zero_grad()

        with torch.enable_grad():

            cov_V, records = get_cov_V_by_trek(F_bin, F, diag_population_cov_V, records=records)

            cov_X = ((cov_V[x_indexes].T)[x_indexes]).T

            loglikelihood = -0.5*torch.slogdet(cov_X)[1]-0.5*torch.trace(torch.mm(torch.inverse(cov_X).to(emp_cov_X.dtype), emp_cov_X))
            nll= -loglikelihood 

            penalty = torch.max(torch.zeros_like(F),torch.abs(F*torch.tensor(F_bin))-0.99*torch.ones_like(F)).pow(2).sum()
            #+ torch.min(torch.zeros_like(F),torch.abs(F)-0.005*torch.ones_like(F)).pow(2).sum()
            loss = nll + penalty*100


        loss.backward()
        optimizer.step()
        
        print(f"iter {i} loss {loss}, nll {nll}")
        print("mse", ((cov_X-emp_cov_X).pow(2)).mean())
        print(f"time {time.time()-t1}")

    print("End")
    print("cov_X", cov_X)
    print("emp_cov_X", emp_cov_X)

    return (F.detach().numpy())*F_bin, nll.detach().numpy()


def get_F_bin_and_varnames_from_dotgraph(graph):
    var_name_ls = []
    for node in graph.get_nodes():
        node_name = node.to_string().split(" ")[0]
        var_name_ls.append(node_name)

    def get_F_bin(graph, var_name_ls):
        num_of_vars= len(var_name_ls)
        F_bin = np.zeros((num_of_vars,num_of_vars))

        for i, Vi in enumerate(var_name_ls):
            for j, Vj in enumerate(var_name_ls):
                if graph.get_subgraph_list()[1].get_edge(Vi,Vj)!=[]:
                    F_bin[i,j] = 1

        return F_bin

    F_bin = get_F_bin(graph, var_name_ls)
    return F_bin, var_name_ls


def add_edge_coefficients_to_graph(graph, F_bin, F, var_name_ls):

    for i in range(F_bin.shape[0]):
        for j in range(F_bin.shape[1]):
            if i!=j:
                if F_bin[i,j]==1:

                    Node_j = var_name_ls[j]
                    Node_i = var_name_ls[i]
                    graph.get_subgraph_list()[1].del_edge(Node_i,Node_j)

                    def get_penwidth(weight):
                        if abs(weight)>0.8:
                            return 5
                        elif abs(weight)>0.6:
                            return 4
                        elif abs(weight)>0.4:
                            return 3
                        elif abs(weight)>0.2:
                            return 2
                        elif abs(weight)>0.1:
                            return 1
                        else:
                            return 0.3

                    def round(x):
                        if abs(x)>0.1:
                            return np.around(x,2)
                        elif abs(x)>0.01:
                            return np.around(x,3)
                        elif abs(x)>0.001:
                            return np.around(x,4)
                        elif abs(x)>0.0001:
                            return np.around(x,5)
                        else:
                            return np.around(x,6)

                    #weight = np.around(params[i,j].detach().numpy(),5)
                    weight = round(F[i,j])
                    color = 'black'
                    penwidth  = get_penwidth(weight)
                    graph.get_subgraph_list()[1].add_edge(pydot.Edge(Node_i, Node_j, fontcolor=color, color=color, penwidth=penwidth, label=weight))

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser("params")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--method", type=str, default='trek')
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--s", type=str, default="multitasking_data") 
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dot_path", type=str, default="/home/export/xinshuad/model1_afterVA_nll2.5094509311433577.dot")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '..')
    from scenarios import dgm_by_scenario
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

