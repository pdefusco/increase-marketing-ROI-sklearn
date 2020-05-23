# Read the fitted model from the file model.pkl
# and define a function that uses the model to
# predict petal width from petal length

import pickle

model = pickle.load(open('model.pkl', 'rb'))

def predict(args):
  petal_length = float(args.get('petal_length'))
  result = model.predict([[petal_length]])
  return result[0][0]
