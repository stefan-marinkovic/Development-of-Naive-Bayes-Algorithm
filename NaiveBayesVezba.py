import pandas as pd
import numpy as np
data=pd.read_csv('data/prehlada.csv')
new_data=pd.read_csv('data/prehlada_novi.csv')

#%% LEARN and PREDICT
def learn(data,class_atr):
    model={}
    _apriori=data[class_atr].value_counts(normalize=True)
    model['_apriori']=_apriori
    
    for atr in data.drop(class_atr,axis=1).columns:
        model[atr]=pd.crosstab(data[atr], data[class_atr],normalize='columns')
        
    return model

def predict(model,new_data):
    pre={}
    for izlazna in model['_apriori'].index:
        pom=1
        for atr in model.keys():
            pom*= model[atr][izlazna] if atr=='_apriori' else model[atr].loc[new_data[atr],izlazna]
        pre[izlazna]=pom
    suma=sum(pre.values())
    for i in pre.keys():
       pre[i]=pre[i]/suma
    return max(pre,key=pre.get),pre
#%%
model=learn(data, 'Prehlada')
for i in range(len(new_data)):
    print(predict(model, new_data.iloc[i]))
    
