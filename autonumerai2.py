# -*- coding: utf-8 -*-
"""autonumerai2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mcoEdqAGQll2dQXB73mJhV1nFqAcP0lz
"""

!cat /proc/cpuinfo

!pip install eli5 numerapi catboost lightgbm xgboost keras_tuner scikit-optimize shap hyperopt

from xgboost.compat import XGBKFold
import numpy as np
import pyarrow.parquet as pq
import eli5
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMRegressor as lgb,plot_importance
from catboost import CatBoostRegressor
from numerapi import NumerAPI
from google.colab import drive
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor as xgbr
import keras_tuner
from skopt import BayesSearchCV
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
import pickle
#import UMAP
counter = 0

class PiEnsembleModel():
    
    def __init__(self,public_id,secret_key,p):
        
        drive.mount('drive')
      
        self.napi = NumerAPI(public_id= public_id, secret_key =secret_key)
        
        # self.curr_round = self.napi.get_current_round()
        self.curr_round = 429
        
        self.p = p

        self.cat_col ='era'

        self.ms = self.p -1 

        self.drop_feat_imp1  = None
        
        self.riskyF = None
        
    def Init(self,trainvalapp= True, test_app= False):
      
        def initpipeline(X,tr = False):
        
            X['era'] = LabelEncoder().fit_transform(X['era']).astype('int64')

            num_col = X.drop(['era','data_type'],axis = 1).columns

            # massively reduce memory

            X[num_col] = X[num_col].astype('float16')

        # take every fourth 
            if tr:
                eraf = np.arange(self.X_era['era'].min(),self.X_era['era'].max(),4)
                X =  X[ X['era'].isin(eraf)
            
            X= X.drop('data_type', axis =1 )
  
            return X

        def liveprep(live):
            
            live['era']=LabelEncoder().fit_transform(live['era']).astype('int64')

            num_col = live.drop(['era','data_type'],axis= 1)

            live[num_col] = live[num_col].astype('float16')

            live = live.drop('data_type',axis =1 )

           # lid = live.reset_index()['id']

            return live


        if trainvalapp:


            self.X_era = pd.read_parquet('/content/drive/My Drive/CURRDATA/'+str(self.curr_round) + '_v3_numerai_training_data.parquet')

            self.X_era=self.X_era.iloc[int( self.X_era.shape[0]/2): self.X_era.shape[0],:]

            self.X_era  = initpipeline(self.X_era,tr=True)

            self.X_era.to_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')      
    
            self.X_v = pd.read_parquet('/content/drive/My Drive/CURRDATA/'+str(self.curr_round) + '_v3_numerai_validation_data.parquet')

# ram criteria can handle half of validation and training 
            
            self.X_v = self.X_v.iloc[int(self.X_v.shape[0]/2):self.X_v.shape[0]]
        
            self.X_v = initpipeline(self.X_v) 

            self.X_v.to_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')

        elif test_app:
            
            self.X_live = pd.read_parquet('/content/drive/My Drive/CURRDATA/'+str(self.curr_round) + '_v3_numerai_live_data.parquet')

            self.X_live= liveprep(self.X_live)

            self.X_live.to_csv('/content/drive/My Drive/CURRDATA'+str(self.curr_round)+'liveera'+str(1)+'.csv')

        self.X_era.head()

    def LoadData(self, trainvalapp=True, test_app = False):

        if trainvalapp:
        
            self.X_era = pd.read_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')
        
            self.X_era= self.X_era.iloc[0:200,:]
        
            self.y_t = self.X_era['target']
        
            self.X_era = self.X_era.drop(['target','id'],axis =1 )
        
            self.X_era['era'] =  self.X_era['era'].astype('int64')
       
            self.X_v = pd.read_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')
        
            self.y_v = self.X_v['target']
        
            self.v_id = self.X_v['id']
        
            self.X_v['era'] = self.X_v['era'].astype('int64')
        
            self.X_v= self.X_v.drop(['target','id'],axis=1)
            
        if test_app:
            self.X_live = pd.read_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'liveera'+str(1)+'.csv')
            
            self.t_id = self.X_live['id']
        
            self.X_live['era'] = self.X_live['era'].astype('int64')
        
            self.X_live= self.X_live.drop('id',axis=1)

        # test neut func 
     #   print('neut test {}'.format(self.NormNeutPred(self.X_v, self.X_v.columns, self.y_v.values.reshape(-1,1),0.4)))

        return self


    def BoostFeatSelect(self,gridg,n_splits =3 ,perc = 0.90,perc2= 0.95):

        st_time = time.time()

   #     n_neighbors = 15
   #     min_dist = 0
   #     n_components = 60

  #      umap= UMAP(n_neighbors= n_neighbors,min_dist =min_dist, n_components = n_components)
#
    #    dt = umap.fit_transform(self.X_era)

    #    umap_feat= [ f for f in range(dt.shape[1])]

  
        initg = {
       # "n_estimators" : 100,
      #  "max_depth" : 5,
        "learning_rate" : 0.01,
        "eval_metric":"rmse"
   #     "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
        }
     #   sub_tss= KFold(n_splits = 5 )

      #  bscv = BayesSearchCV(lg_fu,gridg, cv = sub_tss)
        
      #  bscv.fit(self.X_era,self.y_t)

        def score(gridg):
            
            model = xgbr(**initg)
            model.fit(self.X_era,self.y_t,eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)],
            verbose=False)
              
            Y_pred = model.predict(self.X_v)
            score = np.sqrt(mean_squared_error(self.y_v, Y_pred))
            #print(score)
            return {'loss': score, 'status': STATUS_OK}   


        def opt(trials,gridg):
            
            best = fmin(score,gridg,algo = tpe.suggest, max_evals = 10)

            return best

        trials = Trials()

        optparams = opt(trials,gridg)

        # apply result of hyperopt hyperparameter tuning 
        lg_fu = xgbr(**initg)
        
        lg_fu.set_params(**optparams)

        
        lg_fu.fit(self.X_era,self.y_t, eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)])
   #     ft_name = self.X_era.columns

        ft_imp= lg_fu.feature_importances_

        sort_imp_ind = np.argsort(-ft_imp)

        sorted_ft_name  = self.X_era.columns[sort_imp_ind]

        ft_name1= sorted_ft_name[0:int(perc * len(sorted_ft_name))]

        self.drop_feat_imp1 = self.X_era.drop(ft_name1, axis=1 ).columns
          
        drop_feat= pd.DataFrame()
#        drop_feat2= pd.DataFrame()
        drop_feat['drop_feat1']= self.drop_feat_imp1
 #       drop_feat2['drop_feat2']= self.drop_feat_imp2
        drop_feat.to_csv('/content/drive/My Drive/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)
  #      drop_feat2.to_csv('/content/drive/My Drive/drop_feat2.csv',encoding='utf-8', index=False)
        
        st_time2 = time.time()
        print('Time taken for MDA {}'.format(st_time2-st_time))

      


        return self



     #Return Pearson product-moment correlation coefficients.        
    @staticmethod
    def CorrelationScore(y_true,y_pred) -> np.float32:
        frame=pd.DataFrame()
        frame['true']=y_true
        frame['pred']=y_pred
        return np.corrcoef(frame['true'],frame['pred'])[0,1]
    
    @staticmethod
    def SaveModel(model,model_file_name,t,path=None):
        if t =='c':
            model.save_model(path+model_file_name)
        else:
            pickle.dump(model,open(path+model_file_name+'.pkl',"wb"))
   
    @staticmethod  
    def LoadModel(model_file_name,t,path=None)->object: 
        if t =='c':
            model = CatBoostRegressor().load_model(path+model_file_name)
        else:
            model = pickle.load(open(path+model_file_name+'.pkl',"rb"))
        return model 



    @staticmethod
    def NormNeutPred(df,neut_col, y_pred,proportion) -> np.array:
        
        exposures = df[neut_col].values

        score = ( y_pred  - proportion* exposures.dot(np.linalg.pinv(exposures).dot(y_pred))).reshape( -1,1)
    
        

        score = score/score().std()

        score = MinMaxScaler().fit_transform(score)

        return score



    def FitEnsembleOverEra(self,gridg,Npochs= 1 , sub_splits = 3 , override = False):
        st_time = time.time()
        #pred_c = np.zeros(self.X_era.shape[0])
        
        
        if override:


            self.drop_feat_imp1 = pd.read_csv('/content/drive/My Drive/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values
        
            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)
         #   self.X_era = self.X_era.drop(self.drop_feat_imp2,axis = 1)

            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)
        #    self.X_v = self.X_v.drop(self.drop_feat_imp2,axis = 1)
        
        
        for n in range(Npochs):
        
            initg = {
      #  "n_estimators" : 1000,
    #    "max_depth" : 5,
              "learning_rate" : 0.01,
              "eval_metric":"rmse"
   #     "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
                    }

     #   sub_tss= KFold(n_splits = 5 )

      #  bscv = BayesSearchCV(lg_fu,gridg, cv = sub_tss)
        
      #  bscv.fit(self.X_era,self.y_t)

            def score(gridg):
            
                model = xgbr(**initg)
                model.fit(self.X_era,self.y_t,eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)],
                verbose=False)
              
                Y_pred = model.predict(self.X_v)
                score = np.sqrt(mean_squared_error(self.y_v, Y_pred))
            #print(score)
                return {'loss': score, 'status': STATUS_OK}   



            def opt(trials,gridg):
            
                best = fmin(score,gridg,algo = tpe.suggest, max_evals = 10)

                return best


            trials = Trials()

            optparams = opt(trials,gridg)

        # apply result of hyperopt hyperparameter tuning 
            model_g = xgbr(**initg)
         
            model_g.set_params(**optparams)

            model_g.fit(self.X_era,self.y_t,eval_set = [(self.X_era,self.y_t),(self.X_v,self.y_v)])

            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_t,model_g.predict(self.X_era))))
                

            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_v,model_g.predict(self.X_v))))

          
        loss_curvey = model_g.evals_result()['validation_0']['rmse']

        loss_curveyv = model_g.evals_result()['validation_1']['rmse']

        fig, ax = plt.subplots(2,1)

        ax[0].plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')
          #ax.set_xticks(np.arange(10,110,10))
        # ax.set_yticks(np.arange(0,0.18,0.02)))
        ax[1].plot(np.arange(0, len(loss_curveyv)),loss_curveyv,c='r',label = ' Validation loss RMSE over CV iter')
        
        ax[0].legend()
        ax[1].legend()
      
        plt.show()
          
        self.SaveModel(model_g,'XGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/content/drive/My Drive/numeraimodels/')

        st_time2= time.time()
              
        print('Time taken to tune XGBR model with hyperopt {}'.format(st_time2 - st_time))



    def PredictSubmit(self,model_id,n_splits = 3,FeatImpSelection= True):


        pred_ct= np.zeros(self.X_live.shape[0])

        org = self.X_live.copy()


        for m in range(0,1):
            if FeatImpSelection:
                
                self.ms = m
                
                self.drop_feat_imp1 = pd.read_csv('/content/drive/My Drive/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

       #         self.drop_feat_imp2 = pd.read_csv('/content/drive/My Drive/CURRDATA/PIdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values
                self.X_live = org
                # feature selection dim reductio
              #  tree based 
                self.X_live= self.X_live.drop(self.drop_feat_imp1,axis= 1)
                
        #        self.X_live= self.X_live.drop( self.drop_feat_imp2,axis = 1)


            model_g = self.LoadModel('XGBoostRegressor Model'+' Model_number: '+str(self.ms),'g',path='/content/drive/My Drive/f0/numeraimodels/')

            pred_ct += model_g.predict(self.X_live)/2


        results = pd.DataFrame()

        results['id']=self.t_id

        results['prediction'] = pred_ct 

        results.to_csv('/content/drive/My Drive/'+'ROUND'+str(self.curr_round)+'.csv')

### SET P value in INIT b4 running
public_id = 'enter id '
secret_key = 'sec'
e=PiEnsembleModel(public_id, secret_key,1)

e.Init()

e.LoadData()


gridg = {
        'max_depth':hp.choice('max_depth', np.arange(4, 7, 1, dtype=int)),
     #   'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
      # 'min_child_weight':hp.choice('min_child_weight', np.arange(250, 350, 10, dtype=int)),
      #  'subsample':hp.quniform('subsample', 0.7, 0.9, 0.1),
     #   'eta':hp.quniform('eta', 0.1, 0.3, 0.1),
        
    #    'objective':'reg:squarederror',

    }
e.BoostFeatSelect(gridg)

gridg2 = {
        'max_depth':hp.choice('max_depth', np.arange(4, 7, 1, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
       'min_child_weight':hp.choice('min_child_weight', np.arange(250, 350, 10, dtype=int)),
        'subsample':hp.quniform('subsample', 0.7, 0.9, 0.1),
     #   'eta':hp.quniform('eta', 0.1, 0.3, 0.1),
        
    #    'objective':'reg:squarederror',

    }
e.FitEnsembleOverEra(gridg,override = True)


e

