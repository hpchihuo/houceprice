import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score  
import xgboost as xgb  
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_train = pd.read_csv('C:/Users/Administrator/Desktop/houceprice/train.csv')
df_test = pd.read_csv('C:/Users/Administrator/Desktop/houceprice/test.csv')
alldata = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
df_train.info()

corrmat = train.corr()  
plt.subplots(figsize=(12,9))  
sns.heatmap(corrmat, vmax=0.9, square=True)  

corrmat.loc[:'SalePrice'].order(ascending=False)

sns.distplot(df_train['GrLivArea'], fit=norm)
df_train.plot.scatter(x='TotalBsmtSF', y='SalePrice')
df_train.plot.scatter(x='BsmtFinSF1', y='SalePrice')
sns.boxplot(x='OverallQual', y='SalePrice', data=df_train)
sns.boxplot(x='GarageCars', y='SalePrice', data=df_train)
sns.boxplot(x='FullBath', y='SalePrice', data=df_train)
sns.boxplot(x='Fireplaces', y='SalePrice', data=df_train)
sns.boxplot(x='YearBuilt', y='SalePrice', data=df_train)
#缺失值情况，共1460行数据
df_train.isnull().sum().order(ascending=False)

feature_list = ['GrLivArea'，'TotalBsmtSF'，'BsmtFinSF1'，'OverallQual'，'GarageCars'，'FullBath'，'Fireplaces']


# alldata['PoolQC'].value_counts()
# poolqc = alldata.groupby('PoolQC')['PoolArea'].mean()
# # 查看有PoolArea数据但是没有poolQC的数据  
# poolqcna = alldata[(alldata['PoolQC'].isnull())& (alldata['PoolArea'] != 0)][['PoolQC', 'PoolArea']]
# # 查看无PoolArea数据但是有poolQC的数据  
# poolareana = alldata[(alldata['PoolQC'].notnull())& (alldata['PoolArea']==0)][['PoolQC','PoolArea']]

# a = pd.Series(alldata.columns)
# GarageList = a[a.str.contains('Garage')].values 
# len(alldata[(alldata['GarageArea']==0)&(alldata['GarageCars']==0)])
# len(alldata[(alldata['GarageArea']!=0) & (alldata['GarageCars'].isnull==True)])

def missing_values(alldata):
    alldata_na = pd.DataFrame(alldata.isnull().sum(), columns={'missingNum'})
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(alldata)*100
    alldata_na['existNum'] = len(alldata) - alldata_na['missingNum']
    alldata_na['train_notna'] = len(df_train) - df_train.isnull().sum()
    alldata_na['test_notna'] = alldata_na['existNum'] - alldata_na['train_notna']
    alldata_na['dtype'] = alldata.dtypes
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])  
    alldata_na.set_index('index',inplace=True)  
    return alldata_na  
alldata_na = missing_values(alldata)

poolqcna = alldata[(alldata['PoolQC'].isnull()) & (alldata['PoolArea'] != 0)][['PoolQC', 'PoolArea']]
areamean = alldata.groupby('PoolQC')['PoolArea'].mean()
for i in poolqcna.index:
	v = alldata.loc[i,['PoolArea']].values
	print(type(np.abs(v-areamean)))
	alldata.loc[i,['PoolQC']] = np.abs(v-areamean).astype('float64').argmin()
	print(alldata.loc[i,['PoolQC']])
alldata['PoolQC'] = alldata['PoolQC'].fillna('None')
alldata['PoolArea'] = alldata['PoolArea'].fillna(0)

GarageList = ['GarageCond','GarageFinish','GarageQual']
for garage in GarageList:
	result = alldata[(alldata['GarageType'].notnull() & alldata[garage].isnull())]['GarageType'].unique()[0]
	fill1 = alldata[alldata['GarageType'] == result][garage].mode().values[0]
	index1 = alldata[(alldata['GarageType'].notnull() & alldata[garage].isnull())]['GarageType'].index
	alldata.loc[index1,garage] = fill1
alldata[['GarageCond','GarageFinish','GarageQual','GarageType']] = alldata[['GarageCond','GarageFinish','GarageQual','GarageType']].fillna('None')
alldata[['GarageCars','GarageArea']] = alldata[['GarageCars','GarageArea']].fillna(0)  
alldata['Electrical'] = alldata['Electrical'].fillna( alldata['Electrical'].mode()[0]) 



a = pd.Series(alldata.columns)
BsmtList = a[a.str.contains('Bsmt')].values
#用众数填充
condition = (alldata['BsmtExposure'].isnull() & alldata['BsmtCond'].notnull())#3个
alldata.ix[(condition)] = alldata['BsmtExposure'].mode()[0]
#用相似属性填充
condition1 = (alldata['BsmtExposure'].notnull() & alldata['BsmtCond'].isnull())#3
alldata.ix[(condition1), 'BsmtCond'] = alldata.ix[(condition1), 'BsmtQual']
#用相似属性填充
condition2 = (alldata['BsmtQual'].isnull()) & (alldata['BsmtExposure'].notnull()) # 2个  
alldata.ix[(condition2),'BsmtQual'] = alldata.ix[(condition2),'BsmtCond']  
condition3 = (alldata['BsmtFinType1'].notnull()) & (alldata['BsmtFinType2'].isnull())
alldata.ix[condition3, 'BsmtFinType2'] = 'Unf'
#对数值属性填充0， 对其他属性填充None
allBsmtNa = alldata_na.ix[BsmtList,:]  
allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype']=='object'].index  
allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype']!='object'].index  
alldata[allBsmtNa_obj] =alldata[allBsmtNa_obj].fillna('None')  
alldata[allBsmtNa_flo] = alldata[allBsmtNa_flo].fillna(0)

# MasVnrM = alldata.groupby('MasVnrType')['MasVnrArea']
# MasVnrM['No'] = 0
# MasVnrM = MasVnrM.median()
# alldata[alldata['MasVnrArea']=='None'] = 0
# mtypena = alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
# for i in mtypena.index:  
#     v = alldata.loc[i,['MasVnrArea']].values  
#     alldata.loc[i,['MasVnrType']] = np.abs(v-MasVnrM).astype('float64').argmin()  
alldata.loc[2610,'MasVnrType'] = alldata[((alldata['MasVnrArea'] <= 208) & (alldata['MasVnrArea']>=188))]['MasVnrType'].mode()[0]
alldata['MasVnrType'] = alldata["MasVnrType"].fillna("None")  
alldata['MasVnrArea'] = alldata["MasVnrArea"].fillna(0)


value1 = list(alldata[alldata["MSZoning"].isnull()]["MSSubClass"])
index1 = list(alldata[alldata["MSZoning"].isnull()]["MSSubClass"].index)
for k,v in zip(index1,value1):
	filldata = alldata[alldata['MSSubClass'] == v]["MSZoning"].mode()[0]
	alldata.loc[k, 'MSZoning'] = filldata



#考虑到LotFrontage 与街道连接的线性脚与Neighborhood  房屋附近位置 存在一定的关系 
alldata.ix[(alldata['LotFrontage']=='No'),'LotFrontage'] = 0 
alldata["LotFrontage"] = alldata.groupby("Neighborhood")["LotFrontage"].transform(  
    lambda x: x.fillna(x.median()))

alldata['KitchenQual'] = alldata['KitchenQual'].fillna(alldata['KitchenQual'].mode()[0]) # 用众数填充  
alldata['Exterior1st'] = alldata['Exterior1st'].fillna(alldata['Exterior1st'].mode()[0])  
alldata['Exterior2nd'] = alldata['Exterior2nd'].fillna(alldata['Exterior2nd'].mode()[0])  
alldata["Functional"] = alldata["Functional"].fillna(alldata['Functional'].mode()[0])  
alldata["SaleType"] = alldata["SaleType"].fillna(alldata['SaleType'].mode()[0])  
alldata["Utilities"] = alldata["Utilities"].fillna(alldata['Utilities'].mode()[0])  
alldata[["Fence", "MiscFeature"]] = alldata[["Fence", "MiscFeature"]].fillna('None')  
alldata['FireplaceQu'] = alldata['FireplaceQu'].fillna('None')  
alldata['Alley'] = alldata['Alley'].fillna('None')  


#异常值处理
outliers_id = df_train[(df_train.GrLivArea>4000) & (df_train.SalePrice<200000)].index
alldata = alldata.drop(outliers_id)

#属性构造（重塑）
map1 = {'AllPub':1, 'NoSeWa':0}
map2 = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
map3 = {'Y':1, 'N':0}
alldata = alldata.replace({'CentralAir':map3})
alldata = alldata.replace({'Utilities':map1})
alldata = alldata.replace({'BsmtCond':map2})
alldata = alldata.replace({'GarageQual':map2})
newdata = pd.concat([pd.get_dummies(alldata['MSZoning'],prefix='MSZoning'), alldata], axis=1)
#时间序列处理
newdata.ix[(newdata['YearBuilt']=='No'),['YearBuilt']] = newdata['YearBuilt'].mode()[0]
XHouseAge = 2010 - alldata['YearBuilt'].apply(int)
XHouseAge.name = 'XHouseAge'
newdata = pd.concat([XHouseAge, newdata], axis=1)

train_feature = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','Fireplaces','XHouseAge1','CentralAir','BsmtCond','GarageQual',
'MSZoning_C (all)' ,'MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','MSZoning_No']
for x in train_feature:
	newdata.ix[(newdata[x]=='No'), [x]] = 0
	newdata[x] = newdata[x].apply(int)

#简单函数，规划化，按照比例缩放
numeric_feats = newdata.dtypes[newdata.dtypes != "object"].index  
t = newdata[numeric_feats].quantile(.75) # 取四分之三分位  
use_75_scater = t[t != 0].index  
newdata[use_75_scater] = newdata[use_75_scater]/newdata[use_75_scater].quantile(.75)


train_X = newdata.ix[:1459,train_feature]
test_X = newdata.ix[1460:,train_feature]
Y = df_train.SalePrice.drop(outliers_id)

#对Y进行归一化处理
max1 = Y.max()
min1 = Y.min()
mean1 = Y.mean()
Y = Y.apply(lambda x:x/(max1-min1))

#验证函数
def rmse_cv(model):  
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))  
    return(rmse)  
clf3=xgb.XGBRegressor(colsample_bytree=0.4,  
                 gamma=0.045,  
                 learning_rate=0.07,  
                 max_depth=20,  
                 min_child_weight=1.5,  
                 n_estimators=300,  
                 reg_alpha=0.65,  
                 reg_lambda=0.45,  
                 subsample=0.95)  
clf3.fit(train_X, Y.values)
xgb_preds = np.expm1(clf3.predict(X_test)) 
score3 = rmse_cv(clf3) 
print("\nxgb score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))

xgb_preds = xgb_preds*(max1-min1)
data1 = {'SalePrice':xgb_preds,'Id':list(range(1461, 2920))}
frame1 = pd.DataFrame(data1)
frame1.to_csv('C:/Users/Administrator/Desktop/houceprice/sample.csv')

