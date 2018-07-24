import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from preprocess import load_data_set, prepare_data_set, write_csv, divide_dataset


def main():
    file = 'train.csv'

    try:
        data, header = prepare_data_set(load_data_set(file))
        data = data.astype(np.float)

        sliced_data = data[:, :data.shape[1] - 1]
        target = data[:, data.shape[1] - 1:].astype(np.float)

        data_train, data_test, target_train, target_test = divide_dataset(sliced_data, target)

        print('target min: ', np.min(target))
        print('target max: ', np.max(target))
        print('target median: ', np.median(target))
        print()

        np.set_printoptions(precision=3, linewidth=140, suppress=True)

        print()
        print('regression: ')

        models = {
            # 'linear_regression': linear_model.LinearRegression(),
            # 'linear_ridge': linear_model.Ridge(),
            # 'lasso': linear_model.Lasso(),
            # 'elastic_net': linear_model.ElasticNet(),
            # 'bayesian_ridge': linear_model.BayesianRidge(),
            # 'decision_tree': DecisionTreeRegressor(max_depth=5),
            'random_forest': RandomForestRegressor()
        }

        for name, model in models.items():
            print('model: ', name)

            model.fit(data_train, target_train.ravel())
            scores = cross_val_score(model, data_train, target_train.ravel(), scoring='neg_mean_absolute_error', cv=10)
            print('mean: ', scores.mean())

            model.fit(data_train, target_train.ravel())
            scores = cross_val_score(model, data_train, target_train.ravel(), scoring='neg_mean_absolute_error', cv=10)
            print('neg mean absolute error: ', scores.mean())

            model.fit(data_train, target_train.ravel())
            scores = cross_val_score(model, data_train, target_train.ravel(), scoring='neg_mean_squared_error', cv=10)
            print('neg mean squared error: ', scores.mean())

            model.fit(data_train, target_train.ravel())
            scores = cross_val_score(model, data_train, target_train.ravel(), scoring='neg_median_absolute_error',
                                     cv=10)
            print('neg median absolute error', scores.mean())

            model.fit(data_train, target_train.ravel())
            scores = cross_val_score(model, data_train, target_train.ravel(), scoring='r2', cv=10)
            print('r2', scores.mean())

            predicted = model.predict(data_test)
            output = np.c_[target_test, predicted]

            header.append('predict')

            write_csv('output.csv', output, header)

            print()
    except FileNotFoundError:
        print('cannot open file for reading: ', file)


main()
