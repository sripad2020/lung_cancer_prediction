import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
data=pd.read_csv('cancer patient data sets.csv')
print(data)
print(data.columns)
print(data.isna().sum())
print(data.info())
print(data.describe())

sn.countplot(data['Gender'])
plt.show()

col=data.columns.values
print(col[2:-2])

for i in col[2:-2]:
    sn.barplot(data[i])
    plt.show()

for i in col[2:-2]:
    print(data[i].describe())
    print(data[i].nunique())
    print(data[i].value_counts())
    print(data[i].dtypes)
    print(data[i].shape)
sn.heatmap(data[col[2:-2]].corr())
plt.show()
print(data.Level.value_counts())
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['level']=lab.fit_transform(data['Level'])
from sklearn.feature_selection import SelectKBest
select_k=SelectKBest(k=15)
select_k.fit(data[col[2:-2]],data['level'])
x=select_k.get_feature_names_out()
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
x=minmax.fit_transform(data[x])
y=pd.get_dummies(data['Level'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from keras.models import  Sequential
from keras.layers import Dense,Dropout
import keras.activations,keras.losses,keras.metrics
models=Sequential()
models.add(Dense(units=x_train.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x_train.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_train.shape[1],activation=keras.activations.relu))
models.add(Dropout(0.3))
models.add(Dense(units=x_train.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_train.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_train.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=3,activation=keras.activations.softmax))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_train,y_train,epochs=200,validation_split=0.4)
from sklearn.model_selection import train_test_split
x_trin,x_tst,y_trin,y_tst=train_test_split(x,data['level'])
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_trin,y_trin)
pred=rf.predict(x_tst)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_s=accuracy_score(y_tst,pred)
print('The accuracy score using random forest is:-> ',accuracy_s)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_trin,y_trin)
predic=dt.predict(x_tst)
accuracy_s=accuracy_score(y_tst,predic)
print('The accuracy score using decision tree is:-> ',accuracy_s)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_trin,y_trin)
prediction=lg.predict(x_tst)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_tst,prediction)
print('The accuracy score of logistic regression is:-> ',ac)
plt.scatter(y_tst,prediction,color='green',edgecolors='red')
plt.xlabel('y_test')
plt.ylabel('predicted')
plt.legend()
plt.show()
plt.plot(hist.history['accuracy'],label='training accuracy',marker='o',color='red')
plt.plot(hist.history['val_accuracy'],label='val_accuracy',marker='o',color='darkblue')
plt.title('Training Vs  Validation accuracy')
plt.legend()
plt.show()