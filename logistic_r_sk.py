import numpy as np
np.warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


wine = np.loadtxt('winequality-red.csv', delimiter = ';',skiprows=1)
data = wine[:,0:11]
output_variable = wine[:,11]

x_train, x_test, y_train, y_test = train_test_split(data, output_variable, stratify=output_variable)

#defining parameters for grid search
param_grid = [
  {'C': [0.000000001,0.0000001,0.00001,0.001,0.1,10,1000,100000,10000000]}
]
regr = LogisticRegression()

#Fitting the model on the training set
grid = GridSearchCV(regr,param_grid,cv=5)    
grid.fit(x_train, y_train)    
regr = grid.best_estimator_                   # Best grid
print('\nThe best estimator after grid search is as follows : \n')
print(grid.best_estimator_)

#Making prediction on test set
prediction = regr.predict(x_test)

#Evaluation
confusion_matrix = confusion_matrix(y_test, prediction)
print('\nConfusion matrix : \n',confusion_matrix)
classification_report = classification_report(y_test, prediction)
print('\nClassification report : \n',classification_report)

#Print the misclassified instances
print('Miscalssified samples')
mis = {}
for index,  (predict, actual) in enumerate(zip(prediction, y_test)):
	if predict != actual: 
		if (actual not in mis.keys()):
			mis[actual] = [x_test[index],predict]	

print('\nmisclassified sample : ')
for a,b in mis.items():
	print('actual class : {}, predicted class : {}, data : {}.'.format(a,b[1],b[0]))


