import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from preprocess import load_data_set, prepare_data_set, write_csv


def main():
    file = 'train.csv'

    try:
        data, header = prepare_data_set(load_data_set(file))
        data = data.astype(np.float)

        sliced_data = data[:, :data.shape[1] - 1]
        target = data[:, data.shape[1] - 1:].astype(np.float)

        print('target min: ', np.min(target))
        print('target max: ', np.max(target))
        print('target median: ', np.median(target))
        print()

        np.set_printoptions(precision=3, linewidth=140, suppress=True)

        print()
        print('regression: ')

        models = {
            'linear_regression': linear_model.LinearRegression(),
            'linear_ridge': linear_model.Ridge(),
            'lasso': linear_model.Lasso(),
            'elastic_net': linear_model.ElasticNet(),
            'bayesian_ridge': linear_model.BayesianRidge(),
            'decision_tree': DecisionTreeRegressor(max_depth=5),
            'random_forest': RandomForestRegressor(random_state=1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, alpha=0.01, max_depth=5, loss='lad')
        }

        for name, model in models.items():
            print('model: ', name)

            model.fit(sliced_data, target.ravel())
            scores = cross_val_score(model, sliced_data, target.ravel(), scoring='neg_mean_absolute_error', cv=10)
            print('mean: ', scores.mean())

            model.fit(sliced_data, target.ravel())
            scores = cross_val_score(model, sliced_data, target.ravel(), scoring='neg_mean_absolute_error', cv=10)
            print('neg mean absolute error: ', scores.mean())

            model.fit(sliced_data, target.ravel())
            scores = cross_val_score(model, sliced_data, target.ravel(), scoring='neg_mean_squared_error', cv=10)
            print('neg mean squared error: ', scores.mean())

            model.fit(sliced_data, target.ravel())
            scores = cross_val_score(model, sliced_data, target.ravel(), scoring='neg_median_absolute_error', cv=10)
            print('neg median absolute error', scores.mean())

            model.fit(sliced_data, target.ravel())
            scores = cross_val_score(model, sliced_data, target.ravel(), scoring='r2', cv=10)
            print('r2', scores.mean())

            predicted = model.predict(sliced_data)
            output = np.c_[data, predicted]

            print()

        header.append('predict')

        write_csv('output.csv', output, header)
    except FileNotFoundError:
        print('cannot open file for reading: ', file)


main()
