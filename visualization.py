# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:03:10 2019

@author: C5278763
"""
import pandas as pd

def visualization():
    print("Function is called")
    df=pd.read_csv("data_distinct.csv")
    #print(df.head())
    #groupcount=df.groupby('landscapeid')
   # print(df['landscapeid'].value_counts())
    #print(df['landscapeid'].hist())
    byLandscapeId=df.groupby('landscapeid')['description'].count().reset_index()    
    print(byLandscapeId)
   # byLandscapeId.plot(kind='hist',x='description',y='landscapeid')
    #byLandscapeId.plot(kind='scatter',x='landscapeid',y='description')
    byLandscapeId.plot(kind='kde')
    #byLandscapeId.hist()

visualization()