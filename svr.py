from sklearn.model_selection import train_test_split as t
from sklearn.svm import SVR
import sklearn

import prepare_data as pd


def fit_and_test():
    data, target = pd.read_train()

    train_x, val_x, train_y, val_y = t(data, target, test_size=0.1)

    m = SVR(C=100)
    m.fit(train_x, train_y)
    pred_y = m.predict(val_x)

    outfile = open("valpreds.csv",'w')
    outfile.write("Actual, Predicted, Delta,\n")

    for i in range(len(val_y)):
        delta = val_y[i] - pred_y[i]
        entry = str(val_y[i]) +", "+str(pred_y[i]) +", "+str(delta) +",\n"
        outfile.write(entry)

    outfile.close()
    print("Score on validation")
    print(m.score(val_x, val_y))


def get_predictions():
    data, target = pd.read_train()

    m = SVR()
    m.fit(data, target)

    id, test_data = pd.read_test()

    predictions = m.predict(test_data)

    outfile = open("submissions.csv", 'w')
    entry = "Observation,Energy\n"
    outfile.write(entry)

    for i in range(len(id)):
        entry = id[i] +"," +str(predictions[i]) + "\n"
        outfile.write(entry)

    outfile.close()
    print("Submission file written")


if __name__ == "__main__":
    fit_and_test()
    #get_predictions()