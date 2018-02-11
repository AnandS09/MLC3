from random import shuffle
import numpy as np
import math

data_path = "./MLC3/"


def read_train():
    train_file = data_path + "train.csv"

    first = True
    tf = open(train_file,'r')

    #tr_data =  np.array([])
    #target  = np.array([])
    tr_data = []
    target  = []

    for row in tf:
        if first:
            first = False
            continue

        elems = row.strip().split(',')
        elem_f = [float(x) for x in elems]

        tr_data.append(elem_f[1:-1])
        target.append(elem_f[-1])

    return np.array(tr_data), np.array(target)


def read_test():
    file = data_path + "test.csv"

    first = True
    tf = open(file,'r')

    test_data = []
    id  = []

    for row in tf:
        if first:
            first = False
            continue

        elems = row.strip().split(',')
        data = elems[1:]
        elem_f = [float(x) for x in data]

        test_data.append(elem_f)
        id.append(elems[0])

    return  id, np.array(test_data)


if __name__ == "__main__":
    a, b =read_train()
    print("Hi")



