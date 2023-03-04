class FeatSelect():
    def __init__(self,X_v,y_v):
       
        self.X_v,self.y_v = X_v,y_v;
        self.ms= 1 
        # prevent potential redunancies apply feature selection now 

    def PermImp(self,perc2=0.95):
        
        self.X_v = self.X_v.drop(self.drop_feat_imp1,axis =1 )

        st_time3=time.time()
#init clock
# call Sklearn Permutation Importance
# Select from model

        perminst = PermutationImportance(lg_fu)

        perminst.fit(self.X_v,self.y_v)
#
        ffilter = SelectFromModel(perminst,threshold = 0.05,prefit=True)

       # self.X_era = filter.transform(self.X_era)

        ft_imp2 = perminst.feature_importances_

        sort_imp_ind2 = np.argsort(-ft_imp2)

        sorted_ft_name2= self.X_v.columns[sort_imp_ind2]
        
        ft_name2 = sorted_ft_name2[0:int(perc2*len(sorted_ft_name2))]

        self.drop_feat_imp2 = self.X_v.drop(ft_name2,axis = 1).columns

        drop_feat2= pd.DataFrame()

        drop_feat2['drop_feat1']= self.drop_feat_imp2
       # drop_feat2.to_csv('/content/drive/My Drive/CURRDATA/PIdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)


        st_time4=time.time()

        print('Time taken for PI {}'.format(st_time4-st_time3))
        
