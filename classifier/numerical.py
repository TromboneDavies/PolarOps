
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

N = 5000

genders = np.random.choice(["M","F"],size=N)
heights = np.where(genders=="M",
    np.random.normal(loc=5*12+8,scale=5,size=N),
    np.random.normal(loc=5*12+5,scale=4,size=N))
shoes = np.where(genders=="M",
    np.random.uniform(3,8,size=N),
    np.random.uniform(5,18,size=N))
IQs = np.random.normal(loc=100,scale=10,size=N)

data = pd.DataFrame({
    'gender':genders,
    'height':heights,
    'shoes':shoes,
    'IQ':IQs})
train = data.sample(frac=.8)
test = data.drop(train.index)

model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer=SGD(lr=.001,momentum=.1), loss="binary_crossentropy")
train_inputs = train[['height','shoes','IQ']].to_numpy()
train_labels = np.where(train['gender'].to_numpy() == "M", 0.0, 1.0)
test_inputs = test[['height','shoes','IQ']].to_numpy()
test_labels = np.where(test['gender'].to_numpy() == "M", 0.0, 1.0)

h = model.fit(train_inputs,train_labels,batch_size=5,epochs=10)

preds = model.predict(test_inputs).round()
results = np.isclose(test_labels, preds[:,0])
print("Got {:.2f}% correct.".format(results.sum()/len(results)*100))

