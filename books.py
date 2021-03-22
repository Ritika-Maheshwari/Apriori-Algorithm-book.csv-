# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:05:48 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
#Importing dataset
df=pd.read_csv("G:\\excelr\\data sets\\Book.csv")
df.describe()

frequent_itemsets = apriori(df, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

plt.figure(figsize=(25,10))
plt.bar(x=list(range(1,11)),height=frequent_itemsets.support[1:11]);plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('itemsets');plt.ylabel('support')

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules1=rules.sort_values('lift',ascending=False,inplace=False)

#removing redundancy
def slist(i):
    return (sorted(list(i))) #sort - to sort the string alphabetically

concat=  rules1.antecedents.apply(slist) + rules1.consequents.apply(slist)
concat=concat.apply(sorted)  #sort - to sort the string alphabetically

rule_sets=list(concat) #converting concat to list from series

uni_rule_sets= [list(m) for m in set(tuple(i) for i in rule_sets )] #set- to remove the duplicate elements

index_sets= []
for i in uni_rule_sets:
    index_sets.append(rule_sets.index(i))
    
# getting rules without any redudancy 
rules_no_red = rules.iloc[index_sets,:]

#sorting rules wrt lift associated with them
rules_no_red.sort_values('lift',ascending= False).head(10)

#persons who bought 'Italcook' also bought 'CookBks' and 'ArtBks'
#persons buying 'GeogBks' and 'ChildBks' have bought 'Italcook'
#perosns buying 'CookBks' also buy 'ItalBks' 
# and many more such rules can be made as per the mentioned in rules_no_red and giving offers or discounts to the audeience as per
#the formed rules will yield the better profits