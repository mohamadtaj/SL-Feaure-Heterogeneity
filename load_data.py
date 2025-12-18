import numpy as np
import pandas as pd
import os
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split


def load_dataset (dataset):

    if (dataset == 'thyroid'):
    
        path = './data/thyroid'
        df = pd.read_csv(os.path.join(path,'Thyroid_Diff.csv'))
        df = df.drop(['Response'], axis=1)
        cat_cols = df.columns[1:].tolist()
        df[cat_cols] = df[cat_cols].apply(lambda col: pd.Categorical(col).codes)
        feature_type_dict = {
            'Age': 'numeric',
            'Gender': 'categorical',
            'Smoking': 'categorical',
            'Hx Smoking': 'categorical',
            'Hx Radiothreapy': 'categorical',
            'Thyroid Function': 'categorical',
            'Physical Examination': 'categorical',
            'Adenopathy': 'categorical',
            'Pathology': 'categorical',
            'Focality': 'categorical',
            'Risk': 'categorical',
            'T': 'categorical',
            'N': 'categorical',
            'M': 'categorical',
            'Stage': 'categorical'

        }
        
#--------------------------------------------------------------------------------------------------------------------- 

    elif (dataset == 'glioma'):
    
        path = './data/glioma'
        df = pd.read_csv(os.path.join(path,'glioma.csv'))
        first_col = df.columns[0]
        df = df[[col for col in df.columns if col != first_col] + [first_col]]        
        feature_type_dict = {
            'Gender': 'categorical',
            'Age_at_diagnosis': 'numeric',
            'Race': 'categorical',
            'IDH1': 'categorical',
            'TP53': 'categorical',
            'ATRX': 'categorical',
            'PTEN': 'categorical',
            'EGFR': 'categorical',
            'CIC': 'categorical',
            'MUC16': 'categorical',
            'PIK3CA': 'categorical',
            'NF1': 'categorical',
            'PIK3R1': 'categorical',
            'FUBP1': 'categorical',
            'RB1': 'categorical',
            'NOTCH1': 'categorical',
            'BCOR': 'categorical',
            'CSMD3': 'categorical',
            'SMARCA4': 'categorical',
            'GRIN2A': 'categorical',
            'IDH2': 'categorical',
            'FAT4': 'categorical',
            'PDGFRA': 'categorical'

        }
        
#--------------------------------------------------------------------------------------------------------------------- 
 
    elif (dataset == 'diabetes'):
        path = './data/diabetes'
        df = pd.read_csv(os.path.join(path, 'diabetes.csv'))         
        feature_type_dict = {
            'Pregnancies': 'numeric',
            'Glucose': 'numeric',
            'BloodPressure': 'numeric',
            'SkinThickness': 'numeric',
            'Insulin': 'numeric',
            'BMI': 'numeric',
            'DiabetesPedigreeFunction': 'numeric',
            'Age': 'numeric'

        }
        
#---------------------------------------------------------------------------------------------------------------------   

    elif (dataset == 's_500_num_not_corr'):
        path = './data/s_500_num_not_corr'
        df = pd.read_csv(os.path.join(path, 's_500_num_not_corr.csv')) 
        feature_type_dict = {
            'X1': 'numeric',
            'X2': 'numeric',
            'X3': 'numeric',
            'X4': 'numeric',
            'X5': 'numeric',
            'X6': 'numeric',
            'X7': 'numeric',
            'X8': 'numeric',
            'X9': 'numeric',
            'X10': 'numeric',
            'X11': 'numeric',
            'X12': 'numeric',
            'X13': 'numeric',
            'X14': 'numeric',
            'X15': 'numeric',

        }  
        
#--------------------------------------------------------------------------------------------------------------------- 

    elif (dataset == 's_500_cat_not_corr'):
        path = './data/s_500_cat_not_corr'
        df = pd.read_csv(os.path.join(path, 's_500_cat_not_corr.csv')) 
        feature_type_dict = {
            'X1': 'categorical',
            'X2': 'categorical',
            'X3': 'categorical',
            'X4': 'categorical',
            'X5': 'categorical',
            'X6': 'categorical',
            'X7': 'categorical',
            'X8': 'categorical',
            'X9': 'categorical',
            'X10': 'categorical',
            'X11': 'categorical',
            'X12': 'categorical',
            'X13': 'categorical',
            'X14': 'categorical',
            'X15': 'categorical',

        } 
        
#--------------------------------------------------------------------------------------------------------------------- 

    elif (dataset == 's_500_mix_not_corr'):
        path = './data/s_500_mix_not_corr'
        df = pd.read_csv(os.path.join(path, 's_500_mix_not_corr.csv')) 
        feature_type_dict = {
            'X1': 'numeric',
            'X2': 'numeric',
            'X3': 'numeric',
            'X4': 'numeric',
            'X5': 'numeric',
            'X6': 'numeric',
            'X7': 'numeric',
            'X8': 'numeric',
            'X9': 'categorical',
            'X10': 'categorical',
            'X11': 'categorical',
            'X12': 'categorical',
            'X13': 'categorical',
            'X14': 'categorical',
            'X15': 'categorical',

        } 


#----------------------------------------------------------------------------------------------------------------------  

    elif (dataset == 'gallstone'):
        path = './data/gallstone'
        df = pd.read_csv(os.path.join(path, 'gallstone.csv')) 
        first_col = df.columns[0]
        df = df[[col for col in df.columns if col != first_col] + [first_col]]
        
        cols = df.columns[:-1]

        feature_type_dict = {
            col: ('categorical' if df[col].nunique() < 7 else 'numeric')
            for col in cols
        }

#----------------------------------------------------------------------------------------------------------------------     

    elif (dataset == 'cdc_3class_balanced'):
        path = './data/cdc_3class_balanced'
        df = pd.read_csv(os.path.join(path, 'cdc_3class_balanced.csv')) 
        first_col = df.columns[0]
        df = df[[col for col in df.columns if col != first_col] + [first_col]]
        
        cols = df.columns[:-1]

        feature_type_dict = {
            col: ('categorical' if df[col].nunique() < 15 else 'numeric')
            for col in cols
        }

#----------------------------------------------------------------------------------------------------------------------  

    elif (dataset == 'cdc_binary5050_stratified'):
        path = './data/cdc_binary5050_stratified'
        df = pd.read_csv(os.path.join(path, 'cdc_binary5050_stratified.csv')) 
        first_col = df.columns[0]
        df = df[[col for col in df.columns if col != first_col] + [first_col]]
        
        cols = df.columns[:-1]

        feature_type_dict = {
            col: ('categorical' if df[col].nunique() < 15 else 'numeric')
            for col in cols
        }

        
#--------------------------------------------------------------------------------------------------------------------- 
    return df, feature_type_dict