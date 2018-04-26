import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()

from sklearn.preprocessing import OneHotEncoder
def encoder(arr):
    ohe = OneHotEncoder(sparse=False)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr)  

def encodebins(bins):    
    arr = [[a] for a in range(bins)]
    res = encoder(arr)
    return res
    
if __name__=="__main__":
    arr = encodebins(9)
    a = arr[0]
    print(len(a))
    print(a[1])