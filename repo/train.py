"""Запускать из /opt/airflow/repo: python train.py."""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']


def main():
    train = pd.read_csv('data_train.csv')
    test = pd.read_csv('data_test.csv')
    model = LogisticRegression(max_iter=100_000, random_state=42)
    model.fit(train[FEATURES].values, train['Survived'].values)
    with open('model.pkl', 'wb') as stream:
        pickle.dump(model, stream, protocol=4)
    print(f'Train: {len(train)} rows; test: {len(test)} rows')
    print(f'Train accuracy: {model.score(train[FEATURES].values, train["Survived"].values):.4f}')
    print('Saved model.pkl')


if __name__ == '__main__':
    main()
