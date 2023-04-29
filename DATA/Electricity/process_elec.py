import time
import pandas as pd
import numpy as np
import os
import datetime
import time as timing

####process Electricity#
def timeformat(timestr):
	time = datetime.datetime.strptime(timestr, "%m/%d/%Y %H:%M")
	return str(time)[0:10]

start_time=time.time()
data_path= r'C:\clean\Explain_time_series/Code/DATA/elec'
data = pd.read_csv(os.path.join(data_path,'electric_consumption.csv'))
data['newDate']=data['Date'].apply(timeformat)
cols = list(data.columns[0:])
#groupby newDate, delete date with missing values
dfgb=list(data.groupby(['newDate']))

newdata= pd.DataFrame(columns=cols)
depth=24
for i in range(len(dfgb)):
	if len(dfgb[i][1])==depth:
		newdata = pd.concat([newdata,dfgb[i][1]])
newdata.index=range(len(newdata))
newdata.to_csv(os.path.join(data_path, 'new_elec_consumption.csv'))

#add idx and obtain dataset x and y
newdata_path= r'C:\clean\Explain_time_series/Code/DATA/Processed/elec'
ID=pd.DataFrame()
for i in range(int(len(newdata)/depth)):
    print(i)
    ID = pd.concat([ID, pd.DataFrame([i] * depth)])
ID.index=range(len(ID))
allX=pd.concat([ID, newdata],axis=1)
allY=pd.concat([ID,newdata['Consumption']],axis=1)
allX.columns=['idx']+cols
allY.columns=['idx','target']


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
thirdy.to_csv(os.path.join(newdata_path,'third_y.csv'))
secondy.to_csv(os.path.join(newdata_path,'second_y.csv'))
firsty.to_csv(os.path.join(newdata_path,'first_y.csv'))
allX=allX.drop(['newDate'],axis=1)
allX.to_csv(os.path.join(newdata_path, 'newX_train.csv'), index=False)


