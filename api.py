import pickle

from flask import Flask, request
from flask_restful import Resource, Api


def deserialize_model(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


app = Flask(__name__)
api = Api(app)

model = deserialize_model('random_forest.pickle')


class Predict(Resource):
    required_data = ['month', 'week_day', 'hour', 'season', 'holiday', 'workingday', 'weather', 'temp', 'humidity',
                     'windspeed']

    def post(self, **kwargs):
        json = request.get_json()

        for d in self.required_data:
            if d not in json:
                return {
                    'error': 'missing required data: ' + d
                }, 400

            try:
                if float(json.get(d, 0.0)) is None:
                    return {
                        'error': 'wrong data type for: ' + d
                    }, 400
            except ValueError:
                return {
                           'error': 'wrong data type for: ' + d
                       }, 400

        data_input = [
            [
                json.get('month', 0), json.get('week_day', 0), json.get('hour', 0), json.get('season', 0),
                json.get('holiday', 0), json.get('workingday', 0), json.get('weather', 0), json.get('temp', 0),
                json.get('humidity', 0), json.get('windspeed', 0)
            ]
        ]

        predicted = model.predict(data_input)

        return {
            'predicted': predicted[0]
        }


api.add_resource(Predict, "/ml/predict")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
