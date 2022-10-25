import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split  # для разделение выборки
from sklearn.preprocessing import StandardScaler  # нормировка


name = ['WorldHappiness_Corruption_2015_2020.csv',
        'BigmacPrice.csv',
        'googleplaystore.csv',
        'RUvideos.csv',
        'USvideos.csv', ]

file_path = os.getcwd() + '\\data\\' + name[0]


def read_csv_data():
    data = pd.read_csv(file_path, delimiter=',')

    # print(data[['Country', 'happiness_score', 'gdp_per_capita', 'family', 'health',
    #             'freedom', 'generosity', 'government_trust', 'dystopia_residual',
    #             'continent', 'Year', 'social_support', 'cpi_score']])
    # по умолчанию возрастание
    data = data.sort_values(['happiness_score', 'health'], ascending=[True, True])

    x_data = data[['gdp_per_capita', 'health', 'freedom', 'government_trust']]
    y_data = data.happiness_score

    print(x_data.head())

    return x_data, y_data


def split_on_train_and_test(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)
    return {"train": [x_train, y_train],
            "test": [x_test, y_test],
            }


def main():
    print("Hello World!")


if __name__ == "__main__":
    x, y = read_csv_data()
    data_dict = split_on_train_and_test(x, y)
    print(f"Записей в тренировочной выборке - {data_dict['train'][0].shape}")
    print(f"Записей в тестовой выборке - {data_dict['test'][0].shape}")

    # нормировка данных
    scaler = StandardScaler()
    data_dict['train'][0] = scaler.fit_transform(data_dict['train'][0])
    data_dict['test'][0] = scaler.transform(data_dict['test'][0])

    # Обучим линейную регрессию и подсчитаем её качество на тесте.
    model = LinearRegression()
    model.fit(data_dict['train'][0], data_dict['train'][1])

    y_train_prediction = model.predict(data_dict['train'][0])
    y_test_prediction = model.predict(data_dict['test'][0])

    print('Train MSE: ', mean_squared_error(data_dict['train'][1], y_train_prediction))
    print('Test MSE: ', mean_squared_error(data_dict['test'][1], y_test_prediction))

    print('Train MAE: ', mean_absolute_error(data_dict['train'][1], y_train_prediction))
    print('Test MAE: ', mean_absolute_error(data_dict['test'][1], y_test_prediction))

    # Визуализируем получившиеся веса
    plt.figure(figsize=(20, 8))
    plt.bar(x.columns, model.coef_)
    plt.show()
