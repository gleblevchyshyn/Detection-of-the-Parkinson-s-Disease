from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def correlation(data, threshold):
    cor_col = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                cor_col.add(colname)
    return cor_col


df = pd.read_csv('data/parkinsons.data')


X_train, X_test, y_train, y_test = train_test_split(df.drop(['name', 'status'], axis=1), df['status'], test_size=0.3,
                                                    random_state=13)

features = correlation(X_train, 0.7)
print(features)

X_train.drop(features, axis=1, inplace=True)
X_test.drop(features, axis=1, inplace=True)

scaler = MinMaxScaler((-1, 1))
X_train = scaler.fit_transform(X_train)

model = XGBClassifier()
model.fit(X_train, y_train.values)

predicted = model.predict(scaler.transform(X_test))
print(accuracy_score(y_test.values, predicted)*100)
