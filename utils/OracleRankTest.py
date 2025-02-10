import numpy as np
from DGM.LinearSCM import LinearSCM

class OracleRankTest(object):

    def __init__(self, DGM: LinearSCM):

        self.DGM = DGM

    def __call__(self, pcols, qcols, r):
        rank = self.DGM.cross_covariance_rank_by_idx_in_xvars(pcols, qcols)
        if rank <= r:
            p=1
        else:
            p=0

        return p
    
    def test(self, pcols, qcols, r, alpha):
        rank = self.DGM.cross_covariance_rank_by_idx_in_xvars(pcols, qcols)
        if rank <= r:
            p=1
        else:
            p=0

        fail_to_reject = p>=alpha

        return fail_to_reject
