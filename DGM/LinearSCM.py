import numpy as np
import pandas as pd
from math import sqrt, log
import sys
from itertools import combinations
import os
import copy

class LinearSCM(object):
    def __init__(self, seed=0):
        self.vars = []
        self.xvars = []
        self.lvars = []

        # f_ij: vi->vj, V=F^T@V+e
        self.F = None 
        # variance of error terms
        self.omega = [] 

        self.seed = seed
        np.random.seed(self.seed)

    def copy(self):
        return copy.deepcopy(self)

    def get_causallearn_adj(self):
        adj=np.zeros_like(self.F)
        for i in range(len(self.F)):
            for j in range(len(self.F)):
                if self.F[i,j]!=0:
                    adj[i,j]=-1
                    adj[j,i]=1

        return adj

    def get_causal_order(self):
        import networkx as nx
        adj = self.F!=0
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        causal_order = list(nx.topological_sort(G))

        return causal_order
    

    def random_variance(self):
        return np.random.uniform(0.1, 1)

    def random_coef(self):
        coef = 0
        while abs(coef) < 1:
            coef = np.random.uniform(low=-5, high=5)
            #coef = np.random.uniform(low=-5, high=5)
        return coef

    def expand_F(self, parents):
        n = len(self.vars)
        newF = np.zeros((n, n))
        newF[: (n - 1), : (n - 1)] = self.F  # Padded
        for parent in parents:
            pindex = self.vars.index(parent)
            newF[pindex, n - 1] = self.random_coef()
        return newF

    def remove_variable(self, varname):
        var_index = self.vars.index(varname)
        
        idxset = [i for i in range(len(self.vars))]
        idxset.remove(var_index)

        self.F = self.F[idxset].T[idxset].T
        self.omega = self.omega[0:var_index] + self.omega[var_index+1:]

        self.vars.remove(varname)
        if varname in self.xvars:
            self.xvars.remove(varname)
        if varname in self.lvars: 
            self.lvars.remove(varname)

    def add_variable(self, varname, observed=True):

        self.vars.append(varname)
        if observed:
            self.xvars.append(varname)
        else:
            self.lvars.append(varname)

        self.omega.append(self.random_variance())  # Add noise variance

        if len(self.vars) == 1:
            self.F = np.zeros((1, 1))
        else:
            self.F = self.expand_F(parents=[])

    def add_edge(self, pa, ch):

        assert(pa in self.vars)
        assert(ch in self.vars)

        paindex = self.vars.index(pa)
        chindex = self.vars.index(ch)

        self.F[paindex,chindex]=self.random_coef()

    def remove_edge(self, pa, ch):

        assert(pa in self.vars)
        assert(ch in self.vars)

        paindex = self.vars.index(pa)
        chindex = self.vars.index(ch)

        self.F[paindex,chindex]=0

    def take_interaction(self, another_dgm):
        assert(self.vars==another_dgm.vars)
        result = self.copy()
        result.F = result.F* np.logical_and(result.F!=0, another_dgm.F!=0)

        return result

    def normalize_to_have_unit_variance(self, normalize_var_type='x_and_l'):
        causal_order = self.get_causal_order()
        covariance = self.covariance()
        for i in causal_order:
            if (normalize_var_type=='x_and_l') or (normalize_var_type=='x' and self.vars[i] in self.xvars) or (normalize_var_type=='l' and self.vars[i] in self.lvars): 
                k = sqrt(1/covariance[i][i])

                self.F[:,i] = self.F[:,i]*k
                self.omega[i] = self.omega[i]*k*k
                self.F[i,:] = self.F[i,:]/k


    def generate_data(self, N=100, normalized=True, noise_type='gaussian', leaky_1=1, leaky_2=1):
        np.random.seed(self.seed)
        df_x = pd.DataFrame(columns=self.xvars)
        df_v = pd.DataFrame(columns=self.vars)

        data_v = np.zeros((N,len(self.vars)))
        causal_order = self.get_causal_order()

        for i in causal_order:
            std = sqrt(self.omega[i])
            if noise_type=='gaussian':
                noise = np.random.normal(loc=0, scale=std, size=N)
            elif noise_type=='uniform':
                noise = np.random.uniform(low=-np.sqrt(3)*std, high=np.sqrt(3)*std, size=N)
            elif noise_type=='laplace':
                noise = np.random.laplace(loc=0.0, scale=std/np.sqrt(2), size=N)
            else:
                raise NotImplementedError
            
            data_v[:,i] = (data_v@self.F)[:,i] + noise
            inds_1 = data_v[:,i]>=0
            data_v[inds_1,i] = leaky_1 * data_v[inds_1,i]
            inds_2 = data_v[:,i]<0
            data_v[inds_2,i] = leaky_2 * data_v[inds_2,i]
 
        for i, varname in enumerate(self.vars):
            df_v[varname] = data_v[:, i]
            if varname in self.xvars:
                df_x[varname] = data_v[:, i]

        if normalized:
            df_x=(df_x-df_x.mean())/df_x.std()
            df_v=(df_v-df_v.mean())/df_v.std()

        return df_x, df_v

    def covariance(self):
        n = len(self.vars)
        lamb = np.identity(n) - self.F
        lamb_inv = np.linalg.inv(lamb)
        omega = np.diag(self.omega)
        cov = lamb_inv.T @ omega @ lamb_inv
        return cov

    def cross_covariance_by_varname(self, rowvars, colvars):
        rowindex = [self.vars.index(var) for var in rowvars]
        colindex = [self.vars.index(var) for var in colvars]
        cov = self.covariance()
        return cov[np.ix_(rowindex, colindex)]

    def cross_covariance_rank_by_varname(self, rowvars, colvars):
        cross_cov = self.cross_covariance_by_varname(rowvars, colvars)
        return np.linalg.matrix_rank(cross_cov)
    
    def cross_covariance_rank_by_idx_in_xvars(self, pcols, qcols):

        pcols = [self.vars.index(self.xvars[idx]) for idx in pcols]
        qcols = [self.vars.index(self.xvars[idx]) for idx in qcols]

        cov = self.covariance()
        cross_cov = cov[np.ix_(pcols, qcols)]
        return np.linalg.matrix_rank(cross_cov)

    # Test all possible combinations of subcovariance rank test
    def all_rank_tests(self):
        all_rank_test_result = {}
        n = len(self.xvars)

        combns = []
        for i in range(1, n+1):
            for j in range(i, n+1):
                combns.append((i,j))

        for i, j in combns:
            #print(f"Testing i={i} vs j={j}...")
            Asets = list(combinations(self.xvars, i))
            Bsets = list(combinations(self.xvars, j))
            for A in Asets:
                for B in Bsets:
                    Aset = frozenset(A)
                    Bset = frozenset(B)

                    #if len(Aset.intersection(Bset)) > 0:
                    #    continue
                    #if Aset==Bset:
                    #    continue
                    key = (Aset, Bset)
                    if key in all_rank_test_result:
                        raise Exception
                    #A = sorted(A)
                    #B = sorted(B)
                    rk = self.cross_covariance_rank_by_varname(A, B)
                    all_rank_test_result[key] = rk
        
        return all_rank_test_result