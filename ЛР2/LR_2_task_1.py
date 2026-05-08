import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# Вхідний файл з даними
input_file = 'C:\\Users\\olexi\\OneDrive\\Рабочий стол\\Системи і методи штучного інтелекту\\ЛР2\\income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if len(data) < 15:
            continue
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

print(f"Клас <=50K: {count_class1} зразків")
print(f"Клас  >50K: {count_class2} зразків")

X = np.array(X)

# Кодування рядкових ознак
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розбиття 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# SVM-класифікатор
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=3000))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Метрики
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print("\n=== Метрики якості (LinearSVC) ===")
print(f"Accuracy  (Акуратність): {round(100*acc,  2)}%")
print(f"Precision (Точність):    {round(100*prec, 2)}%")
print(f"Recall    (Повнота):     {round(100*rec,  2)}%")
print(f"F1 Score  (F-міра):      {round(100*f1,   2)}%")
print("\nДетальний звіт:")
print(classification_report(y_test, y_pred,
      target_names=['<=50K', '>50K']))

# Крос-валідація F1
cv_f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print(f"\nF1 (крос-валідація 3-fold): {round(100*cv_f1.mean(), 2)}%")

# Передбачення для тестової точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)
result = label_encoder[-1].inverse_transform(predicted_class)[0]
print(f"\nТестова точка належить до класу: {result}")
