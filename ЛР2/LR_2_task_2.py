import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

input_file = 'C:\\Users\\olexi\\OneDrive\\Рабочий стол\\Системи і методи штучного інтелекту\\ЛР2\\income_data.txt'
X_raw, count1, count2 = [], 0, 0
max_pts = 3000
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count1 >= max_pts and count2 >= max_pts:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if len(data) < 15:
            continue
        if data[-1] == '<=50K' and count1 < max_pts:
            X_raw.append(data); count1 += 1
        if data[-1] == '>50K' and count2 < max_pts:
            X_raw.append(data); count2 += 1

X_raw = np.array(X_raw)
label_encoder, X_enc = [], np.empty(X_raw.shape)
for i, item in enumerate(X_raw[0]):
    if item.isdigit():
        X_enc[:, i] = X_raw[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_enc[:, i] = le.fit_transform(X_raw[:, i])
        label_encoder.append(le)

X = X_enc[:, :-1].astype(int)
y = X_enc[:, -1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

kernels = [
    ('Polynomial (poly, degree=3)', SVC(kernel='poly', degree=3, random_state=0)),
    ('Gaussian RBF (rbf)',          SVC(kernel='rbf',  gamma='scale', random_state=0)),
    ('Sigmoid',                     SVC(kernel='sigmoid', gamma='scale', random_state=0)),
]

print(f"Dataset: {count1} class <=50K + {count2} class >50K")
print("="*55)
for name, clf in kernels:
    clf.fit(X_train, y_train)
    p = clf.predict(X_test)
    print(f"\n--- {name} ---")
    print(f"  Accuracy : {round(100*accuracy_score(y_test,p),2)}%")
    print(f"  Precision: {round(100*precision_score(y_test,p,average='weighted'),2)}%")
    print(f"  Recall   : {round(100*recall_score(y_test,p,average='weighted'),2)}%")
    print(f"  F1 Score : {round(100*f1_score(y_test,p,average='weighted'),2)}%")
