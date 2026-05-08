import numpy as np, matplotlib.pyplot as plt, pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
 
iris_sk = load_iris()
dataset = pd.DataFrame(iris_sk['data'], columns=[
    'sepal-length','sepal-width','petal-length','petal-width'])
dataset['class'] = [iris_sk['target_names'][t] for t in iris_sk['target']]
 
# Крок 3: Розбиття
X = dataset.iloc[:, 0:4].values
y = dataset['class'].values
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)
 
# Крок 4: Побудова та оцінка моделей
models = [('LR', LogisticRegression(solver='lbfgs', max_iter=200)),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]
 
results, names_list = [], []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                  scoring='accuracy')
    results.append(cv_results); names_list.append(name)
    print(f'{name}: {cv_results.mean()*100:.2f}% (+/- {cv_results.std()*100:.2f}%)')
 
# Кроки 6-7: SVM на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
 
# Крок 8: Передбачення нової квітки
X_new = np.array([[5, 2.9, 1, 0.2]])
pred = model.predict(X_new)
print(f'Квітка {X_new[0]} -> сорт: {pred[0]}')
