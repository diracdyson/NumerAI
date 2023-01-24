!pip install eli5 numerapi
import numpy as np
import eli5
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from eli5.sklearn import PermutationImportance
from lightgbm import LGMRegressor as lgb, plot_importance
from catboost import CatBoostRegressor
import numerapi
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
#Init
#LabelEncoder
#OneHotAutoEncoder
#Float32 map
#_neutralize
#FeatImpSelect
#FitEnsembleGridSearchC
# a group of eras must be specified to prevent leakage accross eras
class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self,n_splits):
        super.__init__(n_splits,shuffle=False,random_state=None)
    def split(self,X,y=None, groups):
        X,y, groups = indexable(X,y,groups)
        n_samples = _num_samples(X)
        n_folds = n_splits+ 1
        group_list= np.unique(groups)
        n_groups=(len(group_list))
        if n_folds > n_groups:
            raise ValueError("Cannot have folds ={0} greater")
        indicies = np.arange(n_samples)
        test_size= n_groups//n_folds
        test_starts = range(test_size +n_groups % n_folds, n_groups,test_size)
        test_starts =list(test_starts )[::-1]
        for test in test_starts
            yield(indicies[groups.isin(group_list[:test_start])],indicies[groups.isin(group_list[test_start: test_start + test_size)])
class EnsembleModel():
    def __init__(self,X_train,X_val,y_train,y_val,X_test,public_id,secret_key):
        
        self.X_t=X_train.drop(['id','data_type','target'],axis=1)
        self.y_t= y_train
        self.X_v= X_val.drop(['id','data_type','target'],axis=1)
        self.y_v=y_val
        self.X_live=X_test.drop('data_type',axis=1) 
        self.cat_col='era'
        self.num_col=self.X_t.drop(self.cat_col,axis =1).columns
        if self.X_t.isnull().sum() != 0:
            print("Missing Values X_t")
            break
        elif self.y_t.isnull().sum()!=0:
            print("Missing values y_t")
            break
        elif self.X_v.isnull().sum() !=0:
            print("Missing Values X_v")
            break
        elif self.y_v.isnull().sum() !=0:
            print("Missing values y_v")
            break
        elif self.X_live.isnull().sum() !=0:
            print("Missing Values X_live")
            break
        # activate NumerAI profile connection instance
        self.napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
            
    def LabelEncode(self)->self:
        
        self.X_t[self.cat_col]= LabelEncoder().fit_transform(self.X_t[self.cat_col])
        self.X_v[self.cat_col]= LabelEncoder().fit_transform(self.X_v[self.cat_col]) 
        if not self.X_live.empty:
             self.X_live[self.cat_col]= LabelEncoder().fit_transform(self.X_live[self.cat_col])
        return self
   # reduce memory RAM
   #quantization removes too much info
    def FloatEncode(self,d_type='float32')->self:
       
        self.X_t[self.num_col]=self.X_t[self.num_col].astype(d_type)
        self.X_v[self.num_col]= self.X_v[self.num_col].astype(d_type)
        
        if not self.X_live.empty:
            self.X_live[self.num_col] = self.X_live[self.num_col].astype(d_type)
        return self
   # trees split on accross binary era dimensions
    def OneHotEncode(self)->self:
        self.X_t[self.cat_col]=OneHotEncoder().fit_transform(self.X_t[self.cat_col])
        self.X_v[self.cat_col]=OneHotEncoder().fit_transform(self.X_v[self.cat_col])
        if not self.X_live.empty:
            self.X_live[self.cat_col]= OneHotEncoder().fit_transform(self.X_live[self.cat_col])
        return self 
  # feature discrimination based on feat_imp 
    def FeatImpSelection(self,init_params,grid_ft,n_splits=3,perc=0.95,feat_color='skyblue')->self:
        
        st_time=time.time()
        ftss=TimeSeriesSplit(n_splits=n_splits)
    
        lg_fu= lgb(**init_params)
        gft= GridSearchCV(lg_f,grid_ft,cv=(ftss))
        gft.fit(self.X_t,self.y_t)
    

        lg_fu.set_params(**gft.best_params_)
        lf_fu.fit(self.X_t,self.y_t,feature_name='auto')
        boost = lg_fu.booster_
    
        ft_name1=boost.feature_name()
        ft_name2=self.X_t.columns
    
        if (ft_name1==ft_name2).all():
            ft_name=ft_name2
    #print(ft_name)
        ft_imp=lg_fu.feature_importances_
   # print(ft_imp)
    # select the top 80 percent feature importance
    #col_select = ft_name[0:int(0.8*len(ft_imp))]
    
    #plt.bar(ft_name,ft_imp,color='skyblue',)
        plot_importance(lg_fu,figsize=(50,30),color=feat_color)
        plt.show()
    #plt.xlabel('Features')
    #plt.ylabel('Gini Split based Feature Importance')
    
    
        sort_imp_ind= np.argsort(-ft_imp)
    #print(sort_imp_ind)
        sorted_ft_name= ft_name[sort_imp_ind]
        ft_name_selected= sorted_ft_name[0:int(perc * len(sorted_ft_name))]
    #print(ft_name_selected)

        drop_col=self.X_t.drop(ft_name_selected,axis=1).columns

        self.X_t = self.X_t.drop(drop_col,axis=1)
        self.X_v = self.X_v.drop(drop_col,axis=1)
        if not self.X_live.empty:
            self.X_live= self.X_live.drop(drop_col, axis=1 )


        perminst = PermutationImportance(lg_fu, cv=ftss)
        perminst.fit(self.X_t,self.y_t)

# perm.feature_importances_ attribute is now available, it can be used
# for feature selection - let's e.g. select features which increase
# accuracy by at least 0.05:
        PIFilter = SelectFromModel(perminst, threshold=0.05, prefit=True)
        self.X_t = PIFilter.transform(self.X_t)
        self.X_v= PIFilter.transform(self.X_v)
        if not self.X_live.empty:
            self.X_live=PIFilter.transform(self.X_live)
            
        print('Time taken {}'.format(time.time()-st_time))
        return self
        
    #create ensemble here
# post ensemble model crap 
# neuti
# saving and loading model
# reduce feature exposure aka overfitting
    # post processing
    # this does not account over eras
    @staticmethod
    def NormNeutralizedPred(df,y_pred,proportion) -> np.array():
        exposures = df.values
        score=( y_pred - proportion* exposures.dot(np.linalg.pinv(exposures).dot(y_pred)))
        score = score/score.std()
        score = MinMaxScaler().fit_transform(score)
        return score
   # era based neut
   @staticmethod
    def NormNeutralizedPredGroup(df,neut_col,y_pred,proportion,normalize=True)->np.array():
        if neut_col is None:
            neut_col= []
        unique_eras = df['era'].unique()
        computed=[]
        for u in unique_eras:
            df_era=df[df['era']==u]
            scores=y_pred
            if normalize:
                scores2=[]
                for x in scores.T:
                    x=(scipy.stats.rankdata(x,method ='ordinal' -.5)/len(x)
                    x=scipy.stats.ppf(x)
                    scores2.append(x)
                scores=np.array(scores2).T
            exposures = df_era[neut_col].values
        # ensure it is float32 for memeory and comp efficency 
            scores -= proportion * exposures.dot(np.linalg.pinv(exposures)).dot(scores)
            scores /= scores.std(ddof=0)
            computed.append(scores)
            forecast =np.concatenate(computed)
            forecast= MinMaxScaler().fit_transform(forecast)
        return forecast

    @staticmethod
    def CorrelationScore(y_true,y_pred) -> np.float64:
        return np.corrceof(y_true,y_pred)[0,1]
    
    @staticmethod
    def SaveModel(model,model_file_name,file_ex='.pkl',path=None)
        joblib.dump(model,path+model_file_name+file_ex)
        pass
   
    @staticmethod 
    def LoadModel(model_file_name,file_ex='.pkl',path=None): 
        model=joblib.load(path+model_file_name+file_ex)
        return model 
    
    def SavePredictions(self,y_pred,file_name,model_id,file_ex='.csv',path=None,upload= True ):
        if self.X_live.empty !=0:
            results=pd.DataFrame()
            results['id']= self.X_live['id']
            results['prediction']=y_pred
            results.to_csv(path+file_name+file_ex,index=False)
            if upload:
                submission_id = self.napi.upload_predictions(path+file_name+file_ex, model_id=model_id)
            pass
# declar this crap tmmrw 
# waior ft_imp grid search to occur
#returns dict of folds of ensemble of models 
    def fit(self,init_paramsl,init_paramsc,gridl,gridc,
                               ,Npochs = 1,n_splits= 3,models_app=True,plt=False,neut=True)->dict():
        st_time=time.time()
        cbr=CatBoostRegressor(**init_paramsc)
        lg = lgb(**init_paramsl)
    
        if models_app:
            models=dict()
    
        pred_c=np.zeros(len(self.X_v['era'].values.reshape(-1,1)))
        pred_l=np.zeros(len(self.X_v['era'].values.reshape( -1,1)))
  #  pred_lt=np.zeros( len(X_train['era'].values.reshape( -1,1)))
    
    #k = KFold(N_Fold)
    # dobule split on training 
    # 
        tss=TimeSeriesSplit(n_splits=n_splits)

    
        for i in range(Npochs):
            for _fold, (splitter) in enumerate(tss.split(self.X_t,self.y_t)):
                tr_idx, val_idx = splitter # for cv grid_search 
                cbr.grid_search(gridc, self.X_t.iloc[tr_idx],self.y_t.iloc[tr_idx],cv=tss,plot=plt)
                print("for CatBoostReg @ {} Fold the Training Corr is {}".format(fold_,self.CorrelationScore(self.y_t.iloc[tr_idx],cbr.predict(self.X_t.iloc[tr_idx]))))
                print("for CatBoostReg @ sub {} Fold the Val Corr is {}".format(fold_,self.CorrelationScore(self.y_t.iloc[val_idx],cbr.predict(self.X_t.iloc[val_idx]))))

                glg= GridSearchCV(estimator=lg,param_grid=gridl,cv =tss)
                glg.fit(self.X_t.iloc[tr_idx],self.y_t.iloc[tr_idx])
                
             # grid search updates the parameters of the class instance no worries to update with pointer from dict
           # cbr.set_params
                
       #     
                print("for LightGBMReg @ {} Fold the Training Corr is {}".format(fold_,self.CorrelationScore(self.y_t.iloc[tr_idx],glg.predict(self.X_t.iloc[tr_idx]))))
                print("for LightGBMReg @ sub {} Fold the Val Corr is {}".format(fold_,self.CorrelationScore(self.y_t.iloc[val_idx],glg.predict(self.X_t.iloc[val_idx]))))

                
          
      #  print("for light gbm the Val Corr is {}".format(correlation_score(y_train,glg.predict(X_train))))            
                pred_l+= glg.predict(self.X_v)/n_splits 
                pred_c+= cbr.predict(self.X_v)/n_splits
            #pred_l+= glg.predict(X_test)/N_Fold
            
         #   pred_lt += glg.predict(X_train)
                if models_app:
                    cur_key1='LightGBMReg Model @ TS Fold '+str(_fold)
                    cur_key2='CatBoostReg Model @ TS Fold ' + str(_fold)
                # append to dict for reproduction X_test is too much RAM seperate process VM
                    models[cur_key1]=(glg)
                    models[cur_key2]=(cbr)
    # end of epoch final func protocol 
    
        pred_cl=np.mean([pred_c,pred_l],axis = 0)
        
 #   pred2t=interpscale(pred_lt,0 ,1 )
 #   print(" for light gbm the Training Corr is { }".format(correlation_score(y_train,pred2t)))
        print(" for LightGBMReg the Val Corr is {}".format(self.CorrelationScore(self.y_v,pred_l)))
        print(" for CatBoostReg the Val Cross is {}".format(self.CorrelationScore(self.y_v,pred_c)))
        print(" for bagged ensemble of LightGBMReg + CatBoostReg the Val Cross is {}".format(self.CorrelationScore(self.y_v,pred_cl)))
        

        if neut:
            prop=np.arange(0.05,1,0.05)
        
            predneut_l = [self.CorrelationScore(self.y_v,self.NormNeutralizedPred(self.X_v,pred_l,l)) for l in prop ]
            neutindex_l=np.where(predneut_l==np.max(predneut_l ))
            print(" for LightGBMReg the Neutralized Val Corr is {} and optimal with proportion {}".format(predneut_l[neutindex_l], prop[neutindex_l]))


            predneut_c = [self.CorrelationScore(self.y_v,self.NormNeutralizedPred(self.X_v,pred_c,c)) for c in prop ]
            neutindex_c=np.where(predneut_c==np.max(predneut_c ))
            print(" for CatBoostReg the Neutralized Val Corr is {} and optimal with proportion {}".format(predneut_c[neutindex_c], prop[neutindex_c]))


            predneut_cl = [self.CorrelationScore(self.y_v,self.NormNeutralizedPred(self.X_v,pred_cl,cl)) for cl in prop ]
            neutindex_cl=np.where(predneut_cl==np.max(predneut_cl ))
            print(" for CatBoostReg the Neutralized Val Corr is {} and optimal with proportion {}".format(predneut_cl[neutindex_cl], prop[neutindex_cl]))
            pass
    # decide which model to save afterwards
        if models_app:
            print("Total time for ensemble train_val is {}".format((time.time()-st_time)))
            return models

    def PredictSubmit(self,models,file_name,model_id,n_splits=3,proportion=0.1,save_m=True ):
        if  not self.X_live.empty: 
            pred_c=np.zeros(len(self.X_live['era'].values.reshape(-1,1)))
            pred_l= np.zeros(len(self.X_live['era'].values.reshape(-1,1)))
            for _fold in range(n_splits):
                cur_key1='LightGBMReg Model @ TS Fold '+str(_fold)
                cur_key2='CatBoostReg Model @ TS Fold ' + str(_fold)
            
                model_l = models[cur_key1]
                model_c = models[cur_key2]
                if save_m:
                    self.SaveModel(model_l,'model LGB @ Fold '+str(_fold),path='/numeraimodels')
                    self.SaveModel(model_c,'model CB @ Fold ' +str(_fold),path='/numeraimodels')
                
            
                pred_l += model_l.predict(self.X_live)/n_splits
                pred_c += model_c.predict(self.X_live)/n_splits
        
            pred_cl=np.mean( [pred_l,pred_c],axis=0)
            neutnorm_pred_cl=self.NormNeutralizedPred(self.X_live, pred_cl, proportion )
            self.SavePredictions(neutnorm_pred_cl,file_name,model_id,path='/numeraimodels',upload=False)
            
            



        
