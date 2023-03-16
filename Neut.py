import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn import linear_model


class Neutralization():
    def __init__(self):
        pass
    # remove the trend/stationarity from the non
    # feature exposure is beta w respect to target var
    @staticmethod
    def OLSNeut(endog,preds,proportion=0.15) -> pd.DataFrame():
        
        scores = preds
        exposures = endog.values
        lm = linear_model.Lasso(alpha = 10)
        lm.fit(exposures)
        lm_sample = lm.predict(exposures)
        scores = scores - proportion * lm_sample.reshape(len(scores),1)
        result = pd.DataFrame(scores)
        return result
            
    @staticmethod
    def GLSNeut(endog,preds,proportion =0.15) -> pd.DataFrame():
        
        scores =preds
        exposures =endog.values
        ols_resid = sm.OLS(scores, exposures).fit().resid
        res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
        rho = res_fit.params
        ## orderr by len of parameters
        order = toeplitz(np.arange(len(rho)))
        sigma = rho**order
        #sigma = np.std(preds.reshape( -1,1))
        # can recover sigma from the autocorrelation strucuture built in OLS b_i * x_i + sigma*epsilon 
        gls_model = sm.GLS(scores,exposures, sigma= sigma)
        gls_results = gls_model.fit()
        
        gls_sample = gls_model.predict(exposures)
        scores = scores - proportion * gls_sample.reshape(exposures.shape[0],1)
        result = pd.DataFrame(scores)
        return result
    
