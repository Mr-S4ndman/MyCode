import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json 

from forest import randomforest

if __name__ == "__main__":
    data = np.asarray(pd.read_csv("sdss_redshift.csv"))
    data_new = np.asarray(pd.read_csv("sdss.csv"))
    data_train = data[:4000]
    # data_test = np.asarray(pd.read_csv("sdss.csv"))
    data_test = data[4000:]
    maxdepth=50
    num_features=2
    num_trees=50
    z_train = randomforest(data_train, maxdepth, num_features, num_trees)
    z_train_pred = z_train.predict(data_train[:,:-1])
    y_train = data_train[:,-1]
    
    
    z_test_pred = z_train.predict(data_test)
    y_test = data_test[:,-1]
    
    z_all_pred = z_train.predict(data[:,:-1])
    
    z_new = z_train.predict(data_new)
    
    fig = plt.figure()
    fig.setfigsize=(12,6)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, which='both')
    ax.set_ylim(0, 0.4)
    ax.set_xlim(0, 0.4)
    ax.set_title('Истинное значение-предсказание')
    ax.set_xlabel('Истинное значение')
    ax.set_ylabel('Предсказание')
    plt.scatter(data[:,-1], z_all_pred, s=0.1)
    plt.plot([0, 0.5], [0, 0.5], '-', c = 'r')
    plt.savefig('redshift.png')
    std = {"train": np.std(z_train_pred - y_train), "test": np.std(z_test_pred - y_test)}
    with open('redhsift.json', 'w') as f:
        json.dump(std, f)
    
    # data_new['redshift'] = z_new
    data_new = np.column_stack((data_new, z_new))
    data_new_pd = pd.DataFrame(data_new, columns = ['u','g', 'r', 'i', 'z', 'redshift'])
    data_new_pd.to_csv('sdss_predict.csv')
    
    