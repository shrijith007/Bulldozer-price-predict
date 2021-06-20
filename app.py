import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection  import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import mean_squared_log_error,mean_absolute_error


df=pd.read_csv("./Data/TrainAndValid.csv",low_memory=False,parse_dates=["saledate"])
df.sort_values(by=["saledate"],inplace=True,ascending=True)
df_temp=df.copy()
df_temp["saleyear"]=df_temp.saledate.dt.year
df_temp["salemonth"]=df_temp.saledate.dt.month
df_temp["saleday"]=df_temp.saledate.dt.day
df_temp["saledayofweek"]=df_temp.saledate.dt.dayofweek
df_temp["saledayofyear"]=df_temp.saledate.dt.dayofyear

for label,content in df_temp.items():
        if pd.api.types.is_string_dtype(content):
                df_temp[label]=content.astype("category").cat.as_ordered()


for label,content in df_temp.items():
        if pd.api.types.is_numeric_dtype(content):
                if pd.isnull(content).sum():
                        df_temp[label+'_is_missing']=pd.isnull(content)
                        df_temp[label]=content.fillna(content.median())

for label,content in df_temp.items():
        if not pd.api.types.is_numeric_dtype(content):
                df_temp[label+'_is_missing']=pd.isnull(content)
                df_temp[label]=pd.Categorical(content).codes+1


df_valid=df_temp[df_temp.saleyear ==2012]
df_train=df_temp[df_temp.saleyear !=2012]

x_train,y_train=df_train.drop("SalePrice",axis=1),df_train.SalePrice
x_valid,y_valid=df_valid.drop("SalePrice",axis=1),df_valid.SalePrice


rf_grid={"n_estimators":numpy.arange(10,100,10),
"max_depth":[None,3,5,10],
"min_samples_split":numpy.arange(2,20,2),
"min_samples_leaf":numpy.arange(1,20,2),
"max_features":[0.5,1,"sqrt","auto"],
"max_samples":[10000]}

rs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),param_distributions=rf_grid,n_iter=5,cv=5,verbose=True)

rs_model.fit(x_train,y_train)
train_pred=rs_model.predict(x_train)
valid_pred=rs_model.predict(x_valid)
error=numpy.sqrt(mean_squared_log_error(y_valid,valid_pred))
print(error)