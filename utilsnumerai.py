import pandas as pd 
import numpy as np
import joblib
# implement save and load model
# prediction df->to_csv
# other basic crap
def save_predictions(testing_id,predictions,name_file,file_ex='.csv
                     ',path=None):
    results = pd.DataFrame()
    results['id']= testing_id
    result['results']=predictions
    results.to_csv(path + name_file+file_ex)
    
def save_model(model,model_file_name,file_ex='.pkl',path=None):
    joblib.dump(model,path+file_name+file_ex)

def load_model(file_name,file_ex,path=None):
    model= joblib.load(path+file_name+file_ex)
    return model

def era_encode(df,col):
    lb=LabelEncoder()
    df['era']=  lb.fit_transform(df['era'])
    return df 

def _neutralize(df,preds,proportion):
    scores= preds
    exposures = df.values
    score = scores - proportion* exposures.dot(np.linalg.pinv(exposures).dot(scores))
    return scores/scores.std()




