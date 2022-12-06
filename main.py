import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.model_selection import train_test_split  # для разделение выборки
from sklearn.preprocessing import StandardScaler  # нормировка

from GradientFunction import stochastic_gradient_descent, multi_iteration_sgd
from different_models import *

name = ['WorldHappiness_Corruption_2015_2020.csv',
        'dataset.csv']

file_path = os.getcwd() + '\\data\\' + name[0]


# функции для датасета с счастьем
def WorldHappiness_XYsplit(data):
    """Разделяем на X Y"""
    x_data = data[['Влияние ВВП',
                   'Влияние семьи',
                   'Влияние здоровья',
                   'Влияние свободы',
                   'Влияние щедрости',
                   'Влияния коррупции',]]

    y_data = data[['Рейтинг счастья (0-10)']]
    print(tabulate(data[['Рейтинг счастья (0-10)',
                         'Влияние ВВП','Влияние семьи',
                         'Влияние здоровья','Влияние свободы',
                         'Влияние щедрости','Влияния коррупции',]]
                   , headers='keys', tablefmt='psql'))

    return x_data, y_data


def WorldHappiness_plot(data):
    """Строим зависимости прогнозируемой переменной от параметров"""
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
    # перемешка
    data = data.sample(frac=1).reset_index(drop=True)

    data.rename(columns={'happiness_score': 'Рейтинг счастья (0-10)',
                         'gdp_per_capita': 'Влияние ВВП',
                         'family': 'Влияние семьи',
                         'health': 'Влияние здоровья',
                         'freedom': 'Влияние свободы',
                         'generosity': 'Влияние щедрости',
                         'government_trust': 'Влияния коррупции'}, inplace=True)

    data = data[data['Влияние семьи'] > 0]
    # столбцы с которыми мы будем работать
    part_dataframe = data[['Рейтинг счастья (0-10)', 'Влияние ВВП', 'Влияние семьи', 'Влияние здоровья',
                           'Влияние свободы', 'Влияние щедрости', 'Влияния коррупции']]
    return data, part_dataframe


# визуализируем матрицу кореляции через тепловую и кластерную карту
def do_heatmap(part_dataframe: pd.DataFrame):
    """Тепловая карта"""
    sns.heatmap(part_dataframe.corr(),
                cmap='RdBu_r',  # задаёт цветовую схему
                annot=True,  # рисует значения внутри ячеек
                vmin=-1, vmax=1)  # указывает начало цветовых кодов от -1 до 1.
    plt.show()


def do_clustermap(part_dataframe: pd.DataFrame):
    """Кластерная карта"""
    sns.clustermap(part_dataframe.corr(),
                cmap='RdBu_r',  # задаёт цветовую схему
                annot=True,  # рисует значения внутри ячеек
                vmin=-1, vmax=1)  # указывает начало цветовых кодов от -1 до 1.
    plt.show()


def split_on_train_and_test(X, Y):
    """Записываем в словаь и тестовые и тренировочные данные 0-x 1-y"""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)
    return {"train": [x_train, y_train],
            "test": [x_test, y_test],
            }


if __name__ == "__main__":

    full_data, data_part = read_WorldHappiness()

    # WorldHappiness_plot(full_data)

    do_heatmap(data_part)
    # do_clustermap(data_part)

    x, y = WorldHappiness_XYsplit(full_data)

    data_dict = split_on_train_and_test(x, y)

    # информация о выборке
    print(f"Записей в тренировочной выборке - {data_dict['train'][0].shape}")
    print(f"Записей в тестовой выборке - {data_dict['test'][0].shape}")

    # нормировка данных
    scaler = StandardScaler()
    data_dict['train'][0] = scaler.fit_transform(data_dict['train'][0])
    data_dict['test'][0] = scaler.transform(data_dict['test'][0])

    # модели
    stochastic_gradient_descent(data_part, 'Рейтинг счастья (0-10)')
    multi_iteration_sgd(data_part, 'Рейтинг счастья (0-10)')


    weights = Linear_Regression(data_dict)
    show_weights(x.columns, weights[0])

    weights = Gradient_Linear_Regression(data_dict)
    show_weights(x.columns, weights)

    weights = GridSearchCV_SGD(data_dict)
    show_weights(x.columns, weights)