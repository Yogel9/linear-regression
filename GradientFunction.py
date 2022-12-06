import numpy as np
import pandas as pd
# коэффициент детерминации
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
# Импортируем один из пакетов Matplotlib
import pylab


gradient_step_stah = lambda X, y, w, alpha, ind: w - (alpha * 2.0 / X.shape[0]) * X[ind] * (
                np.dot(X[ind], w) - y[ind])


# стохастический градиентный спуск
def sgd(X, y, X_test, y_test,  w, alpha=1e-4, max_it=10e6):
    # номер итерации
    iter_num = 0
    # ошибки на трейне
    errors = []
    # ошибки на тесте
    errors_test = []
    # r2 на тесте
    r2 = []
    while (iter_num < max_it):
        # выбираем случайный элемент
        ind = np.random.randint(X.shape[0])
        # обновляем веса град спуском
        w = gradient_step_stah(X, y, w, alpha, ind)
        # отображаем каждый %
        if iter_num % (int(max_it/100)) == 0:
            # print(f'Выполнено: {int(iter_num/max_it * 100)}%')
            # mse train
            error = mse(y, np.dot(X, w))
            errors.append(error)
            # print(f'Mse train: {error}')
            # mse test
            error = mse(y_test, np.dot(X_test, w))
            errors_test.append(error)
            # print(f'Mse test: {error}')
            # r2 test
            R = r2_score(y_test, np.dot(X_test, w))
            r2.append(R)
            # print(f'R2:{R}')
        iter_num += 1

    # pylab.plot(errors, label="Mse train")
    # pylab.plot(errors_test, label="Mse test")
    # pylab.plot(r2, label="R2")
    # # pylab.title("")
    # pylab.legend(fontsize=14)
    # pylab.show()

    return w, errors, errors_test, r2


def stochastic_gradient_descent(df: pd.DataFrame, projected_value: str) -> list:
    # # перемешка
    # df = df.sample(frac=1).reset_index(drop=True)
    # train test split
    df_train = df[:200]
    df_test = df[200:]
    print('df_train', df_train)

    print('df_test', df_test)
    # среднее и стандартное отклонение
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    # 0 мат ожидание и 1 дисперсию
    df_train = (df_train - mean) / std

    X_train = df_train.drop(columns=[projected_value]).values
    y_train = df_train[projected_value].values
    df_test = (df_test - mean) / std
    X_test = df_test.drop(columns=[projected_value]).values
    print('df_test', df_test)
    print('x_test', X_test)
    y_test = df_test[projected_value].values

    # выбрали случайный индекс
    ind = np.random.randint(X_train.shape[1])

    # сделали один шаг (w = [1, 1, ...])
    gradient_step_stah(X_train, y_train, np.ones(X_train.shape[1]), 1e-4, ind)
    w, mse_train, mse_test, r2 = sgd(X_train, y_train, X_test, y_test, np.ones(X_train.shape[1]))
    return w


def multi_iteration_sgd(df: pd.DataFrame, projected_value: str):
    # массив результатов
    r2_shuffles = []
    # проверим, зависит ли изначальная перемешка от результата
    for i in range(20):
        print(f'Итерация {i + 1}')
        # перемешка
        df = df.sample(frac=1).reset_index(drop=True)
        # train test split
        df_train = df[:200]
        df_test = df[200:]
        # среднее и стандартное отклонение
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        # 0 мат ожидание и 1 дисперсию
        df_train = (df_train - mean) / std
        X_train = df_train.drop(columns=[projected_value]).values
        y_train = df_train[projected_value].values
        df_test = (df_test - mean) / std
        X_test = df_test.drop(columns=[projected_value]).values
        y_test = df_test[projected_value].values

        w, mse_train, mse_test, r2 = sgd(X_train, y_train, X_test, y_test, np.ones(X_train.shape[1]))
        print(f'Итерация {i + 1} | R^2 = ', r2_score(y_test, np.dot(X_test, w)))
        r2_shuffles.append(r2_score(y_test, np.dot(X_test, w)))
