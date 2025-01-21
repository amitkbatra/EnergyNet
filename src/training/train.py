import gc
import os
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Assuming 'gf' is defined in your notebook as a list of feature names
gf = ['c_syn_cnt:13',
     'c_mss:70',
     'c_mss_max:71',
     'c_mss_min:72',
     'c_win_max:73',
     'c_win_min:74',
     'c_cwin_max:76',
     'c_cwin_min:77',
     'c_cwin_ini:78',
     's_win_scl:90',
     's_mss_max:94',
     's_mss_min:95',
     's_win_max:96',
     's_win_min:97',
     's_cwin_max:99',
     's_cwin_min:100']

data_path_5g = 'Criptominado/crypto5g'
all_files = [f'{data_path_5g}/{files_dir}/log_tcp_temp_complete' for files_dir in os.listdir(data_path_5g)]
all_files = sorted(all_files)
all_files.remove(all_files[0])

df_list = [pd.read_csv(file, sep=' ') for file in all_files]

def tagger(c_ip, s_ip, c_pool, s_pool):
    return 1 if (c_ip in c_pool) and (s_ip in s_pool) else 0

tls_c_pool = [('10.100.200.4'),
          ('10.100.200.3'),
          ('10.100.200.2'),
          ('10.100.200.2'),
          ('10.100.200.2'),
          ('10.100.200.2'),
          ('10.100.200.3'),
          ('10.100.200.4'),
          ('10.100.200.2','10.100.200.4'),
          ('10.100.200.4','10.100.200.2')
]

tls_s_pool = [('149.202.83.171'),
          ('149.202.83.171'),
          ('37.187.95.110'),
          ('94.23.23.52'),
          ('94.23.23.52','149.202.83.171'),
          ('91.121.140.167','149.202.83.171'),
          ('37.187.95.110','91.121.140.167'),
          ('37.187.95.110','91.121.140.167'),
          ('149.202.83.171','37.187.95.110','94.23.23.52','94.23.247.226','91.121.140.167'),
          ('149.202.83.171','37.187.95.110','94.23.23.52','94.23.247.226','91.121.140.167')
]

def tagRows(row, pool_index):
    return tagger(row['#15#c_ip:1'], row['s_ip:15'], tls_c_pool[pool_index], tls_s_pool[pool_index])

for i, df in enumerate(df_list):
    df['tag'] = df.apply(lambda row: tagRows(row, i), axis=1)

df_train = pd.concat([df_list[0],df_list[2],df_list[4],df_list[6]])
df_test = pd.concat([df_list[1],df_list[3],df_list[5],df_list[7]])

def preprocess_data(df_train, df_test, gf):
    standard = StandardScaler().fit(df_train[gf])
    data_transformed = standard.transform(df_train[gf])
    return data_transformed, standard

def baseline_model(input_dim, n_output):
    model = Sequential()
    model.add(Dense(input_dim*4, input_dim=input_dim, activation='relu'))
    model.add(Dense(input_dim*6, activation='relu'))
    model.add(Dense(input_dim*3, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(model, data_transformed, df_train, validation_split, epochs, batch_size, es_patience, es_restore_best_weights, verbose=False, extra_callbacks=[]):
    print("Training model")
    print("Verbose: {}".format(verbose))
    print("Extra callbacks: {}".format(extra_callbacks))

    es = EarlyStopping(monitor='val_loss', mode='min', patience=es_patience, verbose=verbose, restore_best_weights=es_restore_best_weights)
    history = model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=[es] + extra_callbacks, verbose=verbose)

    return model, history

def save_model_keras(model, name):
    model.save(name)

def save_scaler(scaler, name):
    pickle.dump(scaler, open(name, 'wb'))
