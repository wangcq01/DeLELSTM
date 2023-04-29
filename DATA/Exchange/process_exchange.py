import time
import pandas as pd
import numpy as np
import os
import datetime
import time as timing

####process pm25

start_time=time.time()
data_path= r'C:\clean\Explain_time_series/Code/DATA/exchange'
data = pd.read_csv(os.path.join(data_path,'exchange_rate.csv'))

#rolling window data
cols = list(data.columns[1:])
depth=30
X = np.zeros((len(data), depth, len(cols)))
for i, name in enumerate(cols):
    for j in range(depth):
        X[:, j, i] = data[name].shift(depth - j - 1).fillna(method='bfill')
X=X[(depth-1):, :, :]
newX=X.reshape(-1, len(cols))
#newX=X.transpose(2,0,1).reshape(len(cols),-1).transpose(1,0)

#add IDX
allX=pd.DataFrame(newX,columns=cols)
ID=pd.DataFrame()
for i in range(int(len(allX)/depth)):
    print(i)
    ID = pd.concat([ID, pd.DataFrame([i] * depth)])
ID.index=range(len(ID))
newallX=pd.concat([ID, allX],axis=1)
allY=pd.concat([ID,allX['Singapore']],axis=1)
newallX.columns=['idx']+cols
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

newdata_path= r'C:\clean\Explain_time_series/Code/DATA/Processed/exchange'
thirdy.to_csv(os.path.join(newdata_path,'third_y.csv'), index=False)
secondy.to_csv(os.path.join(newdata_path,'second_y.csv'), index=False)
firsty.to_csv(os.path.join(newdata_path,'first_y.csv'), index=False)
newallX.to_csv(os.path.join(newdata_path, 'newX_train.csv'), index=False)


