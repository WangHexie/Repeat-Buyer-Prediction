import pandas as pd
import numpy as np
df = pd.read_csv('data/data_format1/user_info_format1.csv')
count = 0
df.fillna(2,inplace = True)

fileName = 'user_info_format1_NanFix.csv'
df.to_csv("data/data_/" + fileName, encoding='utf-8',index=False)



