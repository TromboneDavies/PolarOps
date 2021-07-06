
from sklearn.model_selection import train_test_split
from prepare import *    # Load/clean data and create model.
import numpy as np
import matplotlib.pyplot as plt



def validate(all_threads, yall, test_size=.2):

    # split into training and test set
    training_threads, test_threads, ytrain, ytest = train_test_split(
        all_threads, yall, test_size=test_size)

    # encode separate training and test matrices
    Xtrain = tokenizer.texts_to_matrix(training_threads, mode='binary')
    Xtest = tokenizer.texts_to_matrix(test_threads, mode='binary')

    model = define_model(numWords)
    histo = model.fit(Xtrain, ytrain, epochs=10, verbose=0)

    return model.predict(Xtest)[:,0].round() == ytest



def validation_hist(numModels=100,title=""):
    accuracies = np.empty(numModels)
    for i in range(numModels):
        print("\nTraining model {}/{}...".format(i+1,numModels))
        results = validate(all_threads, yall)
        accuracies[i] = sum(results)/len(results)*100
    pd.Series(accuracies).hist(density=True, bins=range(0,100,4))
    ax = plt.gca()
    plt.axvline(x=accuracies.mean(),color="red")
    plt.text(x=accuracies.mean()+5,y=.9*ax.get_ylim()[1],
        s="{:.2f}%".format(accuracies.mean()),color="red")
    plt.xlabel("Accuracy (%)")
    plt.title(title)
    return accuracies
        

def validate_one():
    results = validate(all_threads, yall)
    print("\nThe model got {}/{} ({:.2f}%) correct.".format(
        sum(results), len(results), sum(results)/len(results)*100))
