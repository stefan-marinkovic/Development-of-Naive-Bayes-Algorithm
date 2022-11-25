import pandas as pd
import numpy as np
from math import log,exp
from scipy.stats import norm
data=pd.read_csv('data/prehlada.csv')
data['Age']=[14,15,13,45,21,43,67,14,15,13,45,21,43,67,45,32,12,34,56]
new_data=pd.read_csv('data/prehlada_novi.csv')
new_data['Age']=[16,24,54,23,12]
#%% LEARN and PREDICT
def learn(data,class_atr,alfa=5):
    model={}
    _apriori=data[class_atr].value_counts()+alfa
    model['_apriori']=_apriori/_apriori.sum()
    num_col=data.select_dtypes(include=['int','float','complex']).columns
    
    for atr in data.drop(class_atr,axis=1).columns:
        if atr in num_col:
            model[atr]=pd.DataFrame((data[atr].mean(),np.std(data[atr])),index=['mean','std'],columns=[atr])
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
                proba+=log(norm.pdf(new_data[atr], model[atr].loc['mean'], model[atr].loc['std']))
            else:
                proba += log(model[atr][izlazna]) if atr=='_apriori' else log(model[atr].loc[new_data[atr],izlazna])
        pre[izlazna]=proba
    suma=sum({key:exp(value) for key,value in pre.items()}.values())   
    return max(pre,key=pre.get), {key:exp(value)/suma for key,value in pre.items()}
#%%
model=learn(data, 'Prehlada',1)
for i in range(len(new_data)):
    print(predict(model, new_data.iloc[i]))
print(predict(model, new_data.iloc[0]))















lo={}
lo['a']=1
lo['b']=2
exp(lo.values())
pd.Series(lo.items()).apply(lambda i:i[1])
{i:j+1  for i,j in lo.items()}.values().sum()
sum(lo.values())
print("ranicu se")
data.columns[data.columns]
data.select_dtypes(include=['int','float','complex']).columns
pomocna=pd.DataFrame((data['Age'].mean(),np.std(data['Age'])),index=['mean','std'],columns=['sa'])
pomocna.loc['mean']
model['Age']
