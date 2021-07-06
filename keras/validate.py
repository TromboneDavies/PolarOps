
from sklearn.model_selection import train_test_split
from prepare import *    # Load/clean data and create model.

# split into training and test set
training_threads, test_threads, ytrain, ytest = train_test_split(
    all_threads, yall, test_size=.2)

# encode separate training and test matrices
Xtrain = tokenizer.texts_to_matrix(training_threads, mode='binary')
Xtest = tokenizer.texts_to_matrix(test_threads, mode='binary')

histo = model.fit(Xtrain, ytrain, epochs=10, verbose=0)

results = model.predict(Xtest)[:,0].round() == ytest
print("\nThe model got {}/{} ({:.2f}%) correct.".format(
    sum(results), len(results), sum(results)/len(results)*100))

