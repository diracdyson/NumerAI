import numpy as np
import pandas as pd
import seaborn as sns
import scipy 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
from lightgbm import LGMRegressor as lgb, plot_importance
from catboost import CatBoostRegressor
#Init
#LabelEncoder
#OneHotAutoEncoder
#Float32 map
#_neutralize
#FeatImpSelect
#FitEnsembleGridSearchCV
class ensemblemodel():
    def __init__(self,X_train,X_val,y_train,y_val):
        self.X_t=X_train
        self.y_t= y_train
        self.X_v= X_val
        self.y_v=y_val
        if self.X_t.isnull().sum() != 0:
            print("Missing Values X_t")
        elif self.y_t.isnull().sum()!=0:
            print("Missing values y_t")

        elif self.X_v.isnull().sum() !=0:
            print("Missing Values X_v")

        elif self.y_v.isnull().sum() !=0:
            print("Missing values y_v")
    
# declar this crap tmmrw 
# wait for ft_imp grid search to occur

   
