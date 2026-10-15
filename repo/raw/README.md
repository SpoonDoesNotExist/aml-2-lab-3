# Исходные данные Titanic

Резервная копия Kaggle Titanic (891 строка train и 418 строк test).
Скачана 15 октября 2026 года из публичного зеркала:

- https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv
- https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv

Во время выполнения источник catboost.datasets.titanic возвращал HTTP 500.
DAG сначала пробует CatBoost, а при ошибке читает эти исходные CSV и
выполняет ту же обработку пола, возраста и набора признаков.
