import pandas as pd
dataset = pd.read_csv("iris.csv")
X = dataset.iloc[:,:4].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)

#from sklearn.neighbors import KNeighborsClassifier
#KNN = KNeighborsClassifier(n_neighbors=5, p=2, metric = "minkowski")
#KNN.fit(X_train,Y_train)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred,Y_test)
print(cm)

from sklearn.externals import joblib
joblib.dump(classifier,"iris.pkl")