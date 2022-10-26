import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split  # для разделение выборки
from sklearn.preprocessing import StandardScaler  # нормировка

# Регуляризация
name = ['WorldHappiness_Corruption_2015_2020.csv',
        'BigmacPrice.csv',
        'googleplaystore.csv',
        'RUvideos.csv',
        'USvideos.csv', ]

file_path = os.getcwd() + '\\data\\' + name[0]


def WorldHappiness_XYsplit(data):
    """Разделяем на X Y"""
    x_data = data[['Влияние ВВП',
                   'Влияние семьи',
                   'Влияние здоровья',
                   'Влияние свободы',
                   'Влияние щедрости',
                   'Влияния коррупции',]]

    y_data = data[['Рейтинг счастья (0-10)']]
    print(x_data.head())
    return x_data, y_data


def WorldHappiness_plot(data):
    # print(data.columns)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(10, 10))

    ax[0][0].scatter(x='Влияние ВВП', y='Рейтинг счастья (0-10)', data=data)
    ax[0][0].set(xlabel='ВВП', ylabel='Счастье')
    ax[0][1].scatter(x='Влияние семьи', y='Рейтинг счастья (0-10)', data=data)
    ax[0][1].set(xlabel='Семья', ylabel='Счастье')
    ax[0][2].scatter(x='Влияние здоровья', y='Рейтинг счастья (0-10)', data=data)
    ax[0][2].set(xlabel='Здоровье', ylabel='Счастье')
    ax[1][0].scatter(x='Влияние свободы', y='Рейтинг счастья (0-10)', data=data)
    ax[1][0].set(xlabel='Свобода', ylabel='Счастье')
    ax[1][1].scatter(x='Влияние щедрости', y='Рейтинг счастья (0-10)', data=data)
    ax[1][1].set(xlabel='Щедрость', ylabel='Счастье')
    ax[1][2].scatter(x='Влияния коррупции', y='Рейтинг счастья (0-10)', data=data)
    ax[1][2].set(xlabel='Коррупции', ylabel='Счастье')

    plt.show()


def read_WorldHappiness():
    """Считываем и переименовываем/ Отдаем весь датафрейм и его часть"""
    data = pd.read_csv(os.getcwd() + '\\data\\' + name[0])
    data.rename(columns={'happiness_score': 'Рейтинг счастья (0-10)',
                         'gdp_per_capita': 'Влияние ВВП',
                         'family': 'Влияние семьи',
                         'health': 'Влияние здоровья',
                         'freedom': 'Влияние свободы',
                         'generosity': 'Влияние щедрости',
                         'government_trust': 'Влияния коррупции'}, inplace=True)

    # столбцы с которыми мы будем работать
    part_dataframe = data[['Рейтинг счастья (0-10)', 'Влияние ВВП', 'Влияние семьи', 'Влияние здоровья',
                           'Влияние свободы', 'Влияние свободы', 'Влияние щедрости', 'Влияния коррупции']]
    return data, part_dataframe


def do_correlation_matrix(part_dataframe):
    """Матрицей корреляций. """
    sns.heatmap(part_dataframe.corr(),
                cmap='RdBu_r',  # задаёт цветовую схему
                annot=True,  # рисует значения внутри ячеек
                vmin=-1, vmax=1)  # указывает начало цветовых кодов от -1 до 1.
    plt.show()


def split_on_train_and_test(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)
    return {"train": [x_train, y_train],
            "test": [x_test, y_test],
            }


def print_MSE_MAE(y_train, y_test, y_train_prediction, y_test_prediction):
    """вычисляем средневкадратичное и абсолютное ошибки"""
    Train_MSE = mean_squared_error(y_train, y_train_prediction)
    Test_MSE = mean_squared_error(y_test, y_test_prediction)
    Train_MAE = mean_absolute_error(y_train, y_train_prediction)
    Test_MAE = mean_absolute_error(y_test, y_test_prediction)

    print('Train MSE: ', Train_MSE)  # чувствителен к выбросам в выборке
    print('Test MSE: ', Test_MSE)

    print('Train MAE: ', Train_MAE)  # усреднённая сумма модулей разницы между реальным и предсказанным значениями
    print('Test MAE: ', Test_MAE)
    return Train_MSE, Test_MSE, Train_MAE, Test_MAE


def main():
    print("Hello World!")


if __name__ == "__main__":
    result_dict = {}  # сюда запишем результаты
    full_data, data_part = read_WorldHappiness()

    WorldHappiness_plot(full_data)

    do_correlation_matrix(data_part)

    x, y = WorldHappiness_XYsplit(full_data)
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

    print("LinearRegression")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)

    # линейная регрессия с использованием градиентного спуска
    gradient_model = SGDRegressor(tol=.0001, eta0=.01)  # прекращение итерации и скорость обучения
    gradient_model.fit(data_dict['train'][0], data_dict['train'][1])

    y_train_prediction = gradient_model.predict(data_dict['train'][0])
    y_test_prediction = gradient_model.predict(data_dict['test'][0])

    print("SGDLinearRegression")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)

    # # линейная регресси с полиномиальные показателями
    # poly = PolynomialFeatures(2, include_bias=False)  # степень, исключаем x[0]**2
    # poly_df = poly.fit_transform(data_dict['train'][0])
    # # нормировка данных
    # scaled_poly_df = scaler.fit_transform(poly_df)
    # print(f" Количество показателей было: {data_dict['train'][0].shape}")
    # print(f" Количество показателей стало: {scaled_poly_df.shape}")
    # gradient_model.fit(scaled_poly_df, data_dict['train'][1])
    #
    # y_train_prediction = gradient_model.predict(data_dict['train'][0])
    # y_test_prediction = gradient_model.predict(data_dict['test'][0])
    # print("LinearRegression+PolynomialFeatures")
    # print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)

    # Ridge Regression Model
    ridgeReg = Ridge(alpha=10)

    ridgeReg.fit(data_dict['train'][0], data_dict['train'][1])
    y_train_prediction = ridgeReg.predict(data_dict['train'][0])
    y_test_prediction = ridgeReg.predict(data_dict['test'][0])
    print("Ridge")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)

    # Lasso Regression Model
    lasso = Lasso(alpha=1)
    lasso.fit(data_dict['train'][0], data_dict['train'][1])
    y_train_prediction = lasso.predict(data_dict['train'][0])
    y_test_prediction = lasso.predict(data_dict['test'][0])
    print("Laso")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)


    # Визуализируем получившиеся веса
    print(x.columns)
    print(model.coef_[0])
    plt.bar(x.columns, model.coef_[0])
    plt.show()

    print(gradient_model.coef_)
    plt.bar(x.columns, gradient_model.coef_)
    plt.show()

    print(ridgeReg.coef_[0])
    plt.bar(x.columns, ridgeReg.coef_[0])
    plt.show()

    print(lasso.coef_)
    plt.bar(x.columns, lasso.coef_)
    plt.show()
