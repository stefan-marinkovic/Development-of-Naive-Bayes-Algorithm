import pandas as pd
import numpy as np
import sys
from math import log,exp
from scipy.stats import norm
data=pd.read_csv('data/drug.csv')
new_data=data.iloc[-4:]
data=data.iloc[:-4]
#%% LEARN and PREDICT
def learn(data,class_atr,alfa=5):
    model={}
    _apriori=data[class_atr].value_counts()+alfa
    model['_apriori']=_apriori/_apriori.sum()
    num_col=data.select_dtypes(include=['int','float','complex']).columns
    
    for atr in data.drop(class_atr,axis=1).columns:
        if atr in num_col:
            model[atr]=data.groupby(class_atr)[atr].agg(['mean','std'])
        else:
            mat_cont=pd.crosstab(data[atr], data[class_atr])+alfa
            model[atr]=mat_cont/mat_cont.sum(axis=0)
        
    return model

def predict(model,new_data):
    num_col=data.select_dtypes(include=['int','float','complex']).columns
    pre={}
    for izlazna in model['_apriori'].index:
        proba=0
        for atr in model.keys():
            if atr in num_col:
                proba+=log(norm.pdf(new_data[atr], model[atr].loc[izlazna]['mean'], model[atr].loc[izlazna]['std'])+sys.float_info.min)
            else:
                proba += log(model[atr][izlazna]+sys.float_info.min) if atr=='_apriori' else log(model[atr].loc[new_data[atr],izlazna]+sys.float_info.min)
                
        pre[izlazna]=proba
    suma=sum({key:exp(value) for key,value in pre.items()}.values())   
    return max(pre,key=pre.get), {key:exp(value)/suma for key,value in pre.items()}
#%%

model=learn(data, 'Drug',5)
for i in range(len(new_data)):
    droga,procenat=predict(model, new_data.iloc[i,:-1])
    print(droga)
    print(procenat)
    

