from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabulate import tabulate
from prettytable import PrettyTable


def print_MSE_MAE(y_train, y_test, y_train_prediction, y_test_prediction, title=None):
    """вычисляем средневкадратичные и абсолютные ошибки"""
    Train_MSE = mean_squared_error(y_train, y_train_prediction)
    Test_MSE = mean_squared_error(y_test, y_test_prediction)
    Train_MAE = mean_absolute_error(y_train, y_train_prediction)
    Test_MAE = mean_absolute_error(y_test, y_test_prediction)

    table = PrettyTable()
    table.title = title
    table.field_names = ['Train MSE: ', 'Test MSE: ', 'Train MAE: ', 'Test MAE: ']
    table.add_row([Train_MSE, Test_MSE, Train_MAE, Test_MAE])
    print(table.get_string())
    # print('Train MSE: ', Train_MSE)  # чувствителен к выбросам в выборке
    # print('Test MSE: ', Test_MSE)
    #
    # print('Train MAE: ', Train_MAE)  # усреднённая сумма модулей разницы между реальным и предсказанным значениями
    # print('Test MAE: ', Test_MAE)
    return Train_MSE, Test_MSE, Train_MAE, Test_MAE


def Linear_Regression(data_dict):
    """МНК"""""
    model = LinearRegression()
    model.fit(data_dict['train'][0], data_dict['train'][1])

    y_train_prediction = model.predict(data_dict['train'][0])
    y_test_prediction = model.predict(data_dict['test'][0])

    print("LinearRegression")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction,"Linear_Regression")
    return model.coef_


def Gradient_Linear_Regression(data_dict):
    """Градиентный спуск"""
    gradient_model = SGDRegressor(tol=.0001, eta0=.01)  # прекращение итерации и скорость обучения
    gradient_model.fit(data_dict['train'][0], data_dict['train'][1])

    y_train_prediction = gradient_model.predict(data_dict['train'][0])
    y_test_prediction = gradient_model.predict(data_dict['test'][0])

    print("SGDLinearRegression")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction, "Gradient_Linear_Regression")
    return gradient_model.coef_


def GridSearchCV_SGD(data_dict: dict)->list:
    from sklearn.model_selection import GridSearchCV

    grid = {'penalty': ['l1', 'l2'],
            'alpha': [1e-4, 1e-5, 1e-6, 1e-7]}

    reg = SGDRegressor()
    gs = GridSearchCV(reg, grid, cv=5)

    # Обучаем его
    gs.fit(data_dict['train'][0], data_dict['train'][1])
    print(f"Подобранные параметры под SGDRegressor: {gs.best_params_}, {gs.best_score_}")

    # по лучшим параметрам строим модель
    reg = SGDRegressor(alpha=gs.best_params_["alpha"], penalty=gs.best_params_["penalty"])
    reg.fit(data_dict['train'][0], data_dict['train'][1])

    y_train_prediction = reg.predict(data_dict['train'][0])
    y_test_prediction = reg.predict(data_dict['test'][0])

    print("SGDLinearRegression with GridSearchCV")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction,
                  "Gradient_Linear_Regression")
    return reg.coef_

def Polynomial_Linear_Regression(data_dict):
    """линейная регресси с полиномиальные показателями"""
    poly = PolynomialFeatures(2, include_bias=False)  # степень, исключаем x[0]**2
    poly_df = poly.fit_transform(data_dict['train'][0])
    # нормировка данных
    scaled_poly_df = scaler.fit_transform(poly_df)
    print(f" Количество показателей было: {data_dict['train'][0].shape}")
    print(f" Количество показателей стало: {scaled_poly_df.shape}")
    gradient_model.fit(scaled_poly_df, data_dict['train'][1])

    y_train_prediction = gradient_model.predict(data_dict['train'][0])
    y_test_prediction = gradient_model.predict(data_dict['test'][0])
    print("LinearRegression+PolynomialFeatures")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)


def Ridge_Linear_Regression(data_dict):
    """Ridge Regression Model"""
    ridgeReg = Ridge(alpha=10)

    ridgeReg.fit(data_dict['train'][0], data_dict['train'][1])
    y_train_prediction = ridgeReg.predict(data_dict['train'][0])
    y_test_prediction = ridgeReg.predict(data_dict['test'][0])
    print("Ridge")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)
    return ridgeReg.coef_


def Lasso_Linear_Regression(data_dict):
    """Lasso Regression Model"""
    lasso = Lasso(alpha=1)
    lasso.fit(data_dict['train'][0], data_dict['train'][1])
    y_train_prediction = lasso.predict(data_dict['train'][0])
    y_test_prediction = lasso.predict(data_dict['test'][0])
    print("Laso")
    print_MSE_MAE(data_dict['train'][1], data_dict['test'][1], y_train_prediction, y_test_prediction)
    return lasso.coef_


def show_weights(columns_name, coef):
    """Визуализируем получившиеся веса"""
    table = PrettyTable()
    table.title = "Веса"
    table.field_names = columns_name
    table.add_row(coef)
    print(table.get_string())

    # print(columns_name)
    # print(coef)
    plt.bar(columns_name, coef)
    plt.show()
