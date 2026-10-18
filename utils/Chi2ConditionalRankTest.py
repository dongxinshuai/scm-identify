import numpy as np
from pdb import set_trace
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2
from scipy.linalg import eigh
from utils.Chi2RankTest import kcca_modified, recon

class Chi2ConditionalRankTest(object):

    def __init__(self, data_ls, N_scaling=1):

        self.data_ls = data_ls
        for i in range(len(self.data_ls)):
            self.data_ls[i] = self.data_ls[i] - self.data_ls[i].mean(axis=0)
            self.data_ls[i] = self.data_ls[i]/self.data_ls[i].std(axis=0)

        self.N_ls = [data.shape[0] for data in self.data_ls]
        self.N_scaling = N_scaling
        #self.fixed_N = fixed_N

        self.unnormalized_crosscovs_ls = [data.T@data for data in self.data_ls]
        self.cca_cache_dict = {}

        self.test_count = 0

    def get_cachekey(self, pcols_, qcols_):

        pcols = pcols_.copy()
        qcols = qcols_.copy()
        pcols.sort()
        qcols.sort()

        key = ''
        for i in range(self.data_ls[0].shape[1]):
            if i in pcols and i in qcols:
                key = key+str(3)
            elif i in pcols and not i in qcols:
                key = key+str(2)
            elif not i in pcols and i in qcols:
                key = key+str(1)
            else:
                key = key+str(0)

        return key

    def test(self, pcols, qcols, r, alpha):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''
        self.test_count+=1

        cachekey = self.get_cachekey(pcols, qcols)

        if cachekey in self.cca_cache_dict:
            cancorr_ls = self.cca_cache_dict[cachekey]
        else:
            cancorr_ls = []
            for i, data in enumerate(self.data_ls):
                X = data[:, pcols]
                Y = data[:, qcols]

                unnormalized_crosscovs = [self.unnormalized_crosscovs_ls[i][pcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][pcols,:][:,qcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,qcols]]

                try:
                    comps = kcca_modified([X, Y], reg=0.,
                        numCC=None, kernelcca=False, ktype='linear',
                        gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                    cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                    cancorr = cancorr[:,0,1]
                except:
                    print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                    cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                
                #cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr

                cancorr_ls.append(cancorr)

            self.cca_cache_dict[cachekey] = cancorr_ls

        testStat = 0
        p = len(pcols)
        q = len(qcols)
        
        for t, cancorr in enumerate(cancorr_ls):
            testStat_t = 0

            l = cancorr[r:]
            for li in l:
                li = min(li, 1-1e-15)
                testStat_t += log(1)-log(1-li*li)
            ratio = 0
            for i in range(r):
                li = cancorr[i]
                ratio += 1/(li*li)-1

            ratio += self.N_ls[t]*self.N_scaling - r - 0.5*(p+q+1)
            testStat_t = testStat_t * ratio
            testStat += testStat_t

        dfreedom = (p-r) * (q-r) * len(self.data_ls)
        criticalValue = chi2.ppf(1-alpha, dfreedom)
        prob = 1 - chi2.cdf(testStat, dfreedom)
        if_fail_to_reject = testStat<=criticalValue

        # due to numerical errors comparing criticalValue with testStat is more accurate 

        return if_fail_to_reject

    def get_statistics_and_p(self, pcols, qcols, r):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''
        self.test_count+=1

        cachekey = self.get_cachekey(pcols, qcols)

        #if 0:
        if cachekey in self.cca_cache_dict:
            cancorr_ls = self.cca_cache_dict[cachekey]
        else:
            cancorr_ls = []
            for i, data in enumerate(self.data_ls):
                X = data[:, pcols]
                Y = data[:, qcols]

                unnormalized_crosscovs = [self.unnormalized_crosscovs_ls[i][pcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][pcols,:][:,qcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,qcols]]

                try:
                    comps = kcca_modified([X, Y], reg=0.,
                        numCC=None, kernelcca=False, ktype='linear',
                        gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                    cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                    cancorr = cancorr[:,0,1]
                except:
                    print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                    cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                
                #cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr

                cancorr_ls.append(cancorr)

            self.cca_cache_dict[cachekey] = cancorr_ls

        testStat = 0
        p = len(pcols)
        q = len(qcols)
        
        for t, cancorr in enumerate(cancorr_ls):
            testStat_t = 0

            l = cancorr[r:]
            for li in l:
                li = min(li, 1-1e-15)
                #testStat_t += log(1)-log(1-li*li)
                testStat_t += log(1-li*li)

            ratio = 0
            #for i in range(r):
            #    li = cancorr[i]
            #    ratio += 1/(li*li)-1
            #ratio += self.N_ls[t]*self.N_scaling - r - 0.5*(p+q+1)
            ratio = -(self.N_ls[t]*self.N_scaling - 0.5*(p+q+1))

            testStat_t = testStat_t * ratio
            testStat += testStat_t

        dfreedom = (p-r) * (q-r) * len(self.data_ls)
        prob = 1 - chi2.cdf(testStat, dfreedom)

        return testStat, prob
    


class Chi2ConditionalAveragePRankTest(object):

    def __init__(self, data_ls, N_scaling=1):

        self.data_ls = data_ls
        for i in range(len(self.data_ls)):
            self.data_ls[i] = self.data_ls[i] - self.data_ls[i].mean(axis=0)
            self.data_ls[i] = self.data_ls[i]/self.data_ls[i].std(axis=0)

        self.N_ls = [data.shape[0] for data in self.data_ls]
        self.N_scaling = N_scaling
        self.unnormalized_crosscovs_ls = [data.T@data for data in self.data_ls]
        self.cca_cache_dict = {}

    def get_cachekey(self, pcols_, qcols_):

        pcols = pcols_.copy()
        qcols = qcols_.copy()
        pcols.sort()
        qcols.sort()

        key = ''
        for i in range(self.data_ls[0].shape[1]):
            if i in pcols and i in qcols:
                key = key+str(3)
            elif i in pcols and not i in qcols:
                key = key+str(2)
            elif not i in pcols and i in qcols:
                key = key+str(1)
            else:
                key = key+str(0)

        return key

    def get_statistics_and_p(self, pcols, qcols, r):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''

        cachekey = self.get_cachekey(pcols, qcols)

        if 0:
        #if cachekey in self.cca_cache_dict:
            cancorr_ls = self.cca_cache_dict[cachekey]
        else:
            cancorr_ls = []
            for i, data in enumerate(self.data_ls):
                X = data[:, pcols]
                Y = data[:, qcols]

                unnormalized_crosscovs = [self.unnormalized_crosscovs_ls[i][pcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][pcols,:][:,qcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,qcols]]

                '''
                try:
                    comps = kcca_modified([X, Y], reg=0.,
                        numCC=None, kernelcca=False, ktype='linear',
                        gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                    cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                    cancorr = cancorr[:,0,1]
                except:
                    print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                    cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                '''
                cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr

                cancorr_ls.append(cancorr)

            self.cca_cache_dict[cachekey] = cancorr_ls

        testStat_ls = []
        prob_ls = []
        p = len(pcols)
        q = len(qcols)
        

        for t, cancorr in enumerate(cancorr_ls):
            testStat_t = 0

            l = cancorr[r:]
            for li in l:
                li = min(li, 1-1e-15)
                testStat_t += log(1)-log(1-li*li)
            ratio = 0
            for i in range(r):
                li = cancorr[i]
                ratio += 1/(li*li)-1

            ratio += self.N_ls[t]*self.N_scaling - r - 0.5*(p+q+1)
            testStat_t = testStat_t * ratio
            
            dfreedom = (p-r) * (q-r)
            prob_t = 1 - chi2.cdf(testStat_t, dfreedom)
            prob_ls.append(prob_t)
            testStat_ls.append(testStat_t)

        prob_ls = np.array(prob_ls)
        testStat_ls = np.array(testStat_ls)

        return testStat_ls.mean(), prob_ls.mean()
    

class Chi2ConditionalRandomTRankTest(object):

    def __init__(self, data_ls, N_scaling=1):

        self.data_ls = data_ls
        for i in range(len(self.data_ls)):
            self.data_ls[i] = self.data_ls[i] - self.data_ls[i].mean(axis=0)
            self.data_ls[i] = self.data_ls[i]/self.data_ls[i].std(axis=0)

        self.N_ls = [data.shape[0] for data in self.data_ls]
        self.N_scaling = N_scaling
        self.unnormalized_crosscovs_ls = [data.T@data for data in self.data_ls]
        self.cca_cache_dict = {}


    def get_cachekey(self, pcols_, qcols_):

        pcols = pcols_.copy()
        qcols = qcols_.copy()
        pcols.sort()
        qcols.sort()

        key = ''
        for i in range(self.data_ls[0].shape[1]):
            if i in pcols and i in qcols:
                key = key+str(3)
            elif i in pcols and not i in qcols:
                key = key+str(2)
            elif not i in pcols and i in qcols:
                key = key+str(1)
            else:
                key = key+str(0)

        return key

    def test(self, pcols, qcols, r, alpha):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''

        cachekey = self.get_cachekey(pcols, qcols)

        if cachekey in self.cca_cache_dict:
            cancorr_ls = self.cca_cache_dict[cachekey]
        else:
            cancorr_ls = []
            for i, data in enumerate(self.data_ls):
                X = data[:, pcols]
                Y = data[:, qcols]

                unnormalized_crosscovs = [self.unnormalized_crosscovs_ls[i][pcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][pcols,:][:,qcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,qcols]]
                
                try:
                    comps = kcca_modified([X, Y], reg=0.,
                        numCC=None, kernelcca=False, ktype='linear',
                        gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                    cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                    cancorr = cancorr[:,0,1]
                except:
                    print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                    cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                
                #cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                cancorr_ls.append(cancorr)

            self.cca_cache_dict[cachekey] = cancorr_ls

        testStat_ls = []
        prob_ls = []
        p = len(pcols)
        q = len(qcols)
        
        for t, cancorr in enumerate(cancorr_ls):
            testStat_t = 0

            l = cancorr[r:]
            for li in l:
                li = min(li, 1-1e-15)
                testStat_t += log(1)-log(1-li*li)
            ratio = 0
            for i in range(r):
                li = cancorr[i]
                ratio += 1/(li*li)-1

            ratio += self.N_ls[t]*self.N_scaling - r - 0.5*(p+q+1)
            testStat_t = testStat_t * ratio
            
            dfreedom = (p-r) * (q-r)
            prob_t = 1 - chi2.cdf(testStat_t, dfreedom)
            prob_ls.append(prob_t)
            testStat_ls.append(testStat_t)

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-alpha, dfreedom)
        if_fail_to_reject = testStat_ls[-1]<=criticalValue

        # due to numerical errors comparing criticalValue with testStat is more accurate 

        return if_fail_to_reject
        
    def get_statistics_and_p(self, pcols, qcols, r):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''

        cachekey = self.get_cachekey(pcols, qcols)

        if 0:
        #if cachekey in self.cca_cache_dict:
            cancorr_ls = self.cca_cache_dict[cachekey]
        else:
            cancorr_ls = []
            for i, data in enumerate(self.data_ls):
                X = data[:, pcols]
                Y = data[:, qcols]

                unnormalized_crosscovs = [self.unnormalized_crosscovs_ls[i][pcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][pcols,:][:,qcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,pcols], \
                                          self.unnormalized_crosscovs_ls[i][qcols,:][:,qcols]]
                
                try:
                    comps = kcca_modified([X, Y], reg=0.,
                        numCC=None, kernelcca=False, ktype='linear',
                        gausigma=1.0, degree=2, crosscovs = unnormalized_crosscovs)

                    cancorr, _, _ = recon([X,Y], comps, kernelcca=False)
                    cancorr = cancorr[:,0,1]
                except:
                    print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                    cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                
                #cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr
                cancorr_ls.append(cancorr)

            self.cca_cache_dict[cachekey] = cancorr_ls

        testStat_ls = []
        prob_ls = []
        p = len(pcols)
        q = len(qcols)
        
        for t, cancorr in enumerate(cancorr_ls):
            testStat_t = 0

            l = cancorr[r:]
            for li in l:
                li = min(li, 1-1e-15)
                testStat_t += log(1)-log(1-li*li)
            ratio = 0
            for i in range(r):
                li = cancorr[i]
                ratio += 1/(li*li)-1

            ratio += self.N_ls[t]*self.N_scaling - r - 0.5*(p+q+1)
            testStat_t = testStat_t * ratio
            
            dfreedom = (p-r) * (q-r)
            prob_t = 1 - chi2.cdf(testStat_t, dfreedom)
            prob_ls.append(prob_t)
            testStat_ls.append(testStat_t)

        return testStat_ls[-1], prob_ls[-1]