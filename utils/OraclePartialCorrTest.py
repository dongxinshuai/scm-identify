import numpy as np
from DGM.LinearSCM import LinearSCM
from causallearn.utils.cit import CIT_Base
NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"

class OraclePartialCorrTest(CIT_Base):

    def __init__(self, DGM: LinearSCM, fake_data, **kwargs):
        super().__init__(fake_data, **kwargs)
        self.check_cache_method_consistent('OraclePartialCorrTest', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()


        self.DGM = DGM

    def __call__(self, X, Y, condition_set=None):
                 
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]

        pcols = list(set.union(set(Xs),set(condition_set)))
        qcols = list(set.union(set(Ys),set(condition_set)))

        rank = self.DGM.cross_covariance_rank_by_idx_in_xvars(pcols, qcols)

        if rank <= len(set(condition_set)):
            p=1
        else:
            p=0

        self.pvalue_cache[cache_key] = p

        return p
