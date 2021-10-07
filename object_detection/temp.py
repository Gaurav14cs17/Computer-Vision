file_path = "p1.txt"
import csv
import os
import csv


def writeCsvFile(fname, data, *args, **kwargs):
    mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)


mydat = [['loss', 'accuracy', 'val_loss', 'val_accuracy']]

with open(file_path, 'r') as f:
    data = f.readlines()
    for x in data:
        x = x.split('-')
        if len(x) < 4:
            continue
        loss = x[0].split(":")[1]
        accuracy = x[1].split(":")[1]
        val_loss = x[2].split(":")[1]
        val_accuracy = x[3].split(":")[1].split('\n')[0]
        daat = [float(loss), float(accuracy), float(val_loss), float(val_accuracy)]
        mydat.append(daat)

# mydat = tuple(mydat)
# writeCsvFile(r'test.csv', mydat)

import pandas as pd

data = pd.read_csv('test.csv')
print(data.head(10))
