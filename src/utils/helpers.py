import gzip
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

def show_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def get_metrics(Y, predictions):
    if len(Y.unique()) > 2:
        print('F1:',f1_score(Y, predictions, average='macro'))
    else:
        print('F1:',f1_score(Y, predictions))
    print('Accuracy:',accuracy_score(Y, predictions))
    print(confusion_matrix(Y, predictions))

def gzip_model(model_path):
    with open(model_path, 'rb') as f_in, open(f"{model_path}.gz", 'wb') as f_out:
        f_out.write(gzip.compress(f_in.read()))
    os.remove(model_path)
    return f"{model_path}.gz"

def unzip_model(model_path):
    with gzip.open(model_path, 'rb') as f_in, open(model_path[:-3], 'wb') as f_out:
        f_out.write(f_in.read())
    return model_path[:-3]
