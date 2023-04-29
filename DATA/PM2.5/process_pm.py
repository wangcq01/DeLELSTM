import time
import pandas as pd
import numpy as np
import os
import datetime
import time as timing

####process pm25

start_time=time.time()
data_path= r'C:\clean\Explain_time_series/Code/DATA/pm'
data = pd.read_csv(os.path.join(data_path,'PRSA_data_2010.1.1-2014.12.31.csv'))
#add IDX
depth=24
cols = list(data.columns[0:])

data['pm2.5'] = data['pm2.5'].fillna(method='ffill').fillna(method='bfill')
data.loc[data['cbwd']=='cv','cbwd'] = 1
data.loc[data['cbwd']=='NE','cbwd'] = 2
data.loc[data['cbwd']=='NW','cbwd'] = 3
data.loc[data['cbwd']=='SE','cbwd'] = 4
ID=pd.DataFrame()
for i in range(int(len(data)/depth)):
    print(i)
    ID = pd.concat([ID, pd.DataFrame([i] * depth)])
ID.index=range(len(ID))
allX=pd.concat([ID, data],axis=1)
allY=pd.concat([ID,data['pm2.5']],axis=1)
allX.columns=['idx']+cols
allY.columns=['idx','target']


##process y
group_df=list(allY.groupby(['idx']))
firsty=pd.DataFrame() #predict using first data
secondy=pd.DataFrame() #predict y using first two data
thirdy=pd.DataFrame() #predict y using first three data
for i in range(len(group_df)):
	print(i)
	group_df[i][1].index=range(len(group_df[i][1]))
	first=group_df[i][1].drop(index=[0])
	second = group_df[i][1].drop(index=[0,1])
	third=group_df[i][1].drop(index=[0,1,2])
	firsty=pd.concat([firsty,first])
	secondy=pd.concat([secondy,second])
	thirdy=pd.concat([thirdy, third])

firsty.index=range(len(firsty))
secondy.index=range(len(secondy))
thirdy.index=range(len(thirdy))

newdata_path= r'C:\clean\Explain_time_series/Code/DATA/Processed/pm'
thirdy.to_csv(os.path.join(newdata_path,'third_y.csv'), index=False)
secondy.to_csv(os.path.join(newdata_path,'second_y.csv'), index=False)
firsty.to_csv(os.path.join(newdata_path,'first_y.csv'), index=False)
allX.to_csv(os.path.join(newdata_path, 'newX_train.csv'), index=False)


