import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import get_model
import keras
import warnings
warnings.filterwarnings('ignore')

def making_input(games_count, dataset):

    tmp_era = [
        'WLS_changed', 'INN2_teampit',
        'BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit', 'HR_pit', 'SB_pit',
        'BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit', 'INN2_sp', 'ER_sp', 'PE'
    ]

    tmp_avg = [
        'WLS_changed',
        'AB_bat', 'RBI', 'RUN', 'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat',
        'SB_bat', 'BB_bat', 'KK_bat', 'GD_bat', 'ERR', 'LOB',
        #         'type1', 'type2', 'type3', 'type4',
        'PE', '3hal'
    ]
    input_data_era = []
    input_data_avg = []
    labels_AVG = []
    labels_ERA = []
    for i in ['LG', 'OB', 'HH', 'NC', 'HT', 'SS', 'SK', 'KT', 'WO', 'LT']:
        for j in [(20170101, 20171231), (20180101, 20181231), (20190101, 20191231), (20200101, 20201231)]:
            data = dataset[(dataset['T_ID'] == i) & (dataset['GDAY_DS'] < j[1]) & (dataset['GDAY_DS'] > j[0])]
            for k in range(len(data) - 60):
                era_features = data[tmp_era].iloc[k:k + 30, ].values
                avg_features = data[tmp_avg].iloc[k:k + 30, ].values
                era_features = np.append(era_features, data.iloc[k + 30:k + 60, ]['WLS_changed'].values.reshape(30, -1), axis=1)
                avg_features = np.append(avg_features, data.iloc[k + 30:k + 60, ]['WLS_changed'].values.reshape(30, -1), axis=1)

                input_data_era.append(era_features)
                input_data_avg.append(avg_features)

                labels_AVG.append(data[['HIT_bat']].iloc[k + 30:k + 30 + games_count, ].sum().values[0] /
                                  data[['AB_bat']].iloc[k + 30:k + 30 + games_count, ].sum().values[0])
                labels_ERA.append((data[['ER_teampit']].iloc[k + 30:k + 30 + games_count, ].sum().values[0]) / (
                (data[['INN2_teampit']].iloc[k + 30:k + 30 + games_count, ].sum().values[0] / 3)) * 9)
    return np.array(input_data_era), np.array(labels_ERA), np.array(input_data_avg), np.array(labels_AVG), tmp_era, tmp_avg

def Min_Max_scaling_3D(X_train, y_train, X_test, y_test):
    print('MinMaxScaling')
    X_train = np.delete(X_train, -1, axis=2)
    X_test = np.delete(X_test, -1, axis=2)

    max_ls = X_train[0][0].copy()
    min_ls = X_train[0][0].copy()

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                if X_train[i][j][k] > max_ls[k]:
                    max_ls[k] = X_train[i][j][k]

                if X_train[i][j][k] < min_ls[k]:
                    min_ls[k] = X_train[i][j][k]

    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                X_train_std[i][j][k] = (X_train[i][j][k] - min_ls[k]) / (max_ls[k] - min_ls[k])

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            for k in range(X_test.shape[2]):
                X_test_std[i][j][k] = (X_test[i][j][k] - min_ls[k]) / (max_ls[k] - min_ls[k])
    return X_train_std, X_test_std

def train(X_input, y_input, types):
    print('start train')
    model, early = get_model(types)
    model.fit(X_input, y_input,
              batch_size=128,
              epochs=5,
              verbose=1,
              callbacks=[early],
              validation_split=0.1)
    model.save(f'{types}_model.h5')

def predict(types, predict_data):
    print('start predict')
    loaded = keras.models.load_model(f'{types}_model.h5')
    result = loaded.predict(predict_data)
    return result

def for_lgb(X_test_era, y_test_era, y_predict_era,  X_test_avg, y_test_avg, y_predict_avg, era_columns, avg_columns):
    print('start lgb_make')
    df_pred = pd.DataFrame()
    df_pred['ERA_LSTM_pred'] = y_predict_era
    df_pred['ERA_NEXT'] = y_test_era
    df_pred['AVG_LSTM_pred'] = y_predict_avg
    df_pred['AVG_NEXT'] = y_test_avg

    X_era = np.zeros(X_test_era.shape)

    for i in range(X_test_era.shape[0]):
        for j in range(X_test_era.shape[1]):
            for k in range(X_test_era.shape[2]):
                X_era[i][j][k] = X_test_era[i][j][k]
            X_era[i][j][-1] = X_test_era[i][j][-1]

    X_avg = np.zeros(X_test_avg.shape)
    for i in range(X_test_avg.shape[0]):
        for j in range(X_test_avg.shape[1]):
            for k in range(X_test_avg.shape[2]):
                X_avg[i][j][k] = X_test_avg[i][j][k]
            X_avg[i][j][-1] = X_test_avg[i][j][-1]

    dataset_era = X_era.sum(axis=1)
    dataset_avg = X_avg.sum(axis=1)
    df_era = pd.DataFrame(dataset_era, columns=era_columns + ['NEXT_WLS'])
    df_avg = pd.DataFrame(dataset_avg, columns=avg_columns + ['NEXT_WLS'])
    df_era['ERA_pred'] = df_pred['ERA_LSTM_pred']
    df_avg['AVG_pred'] = df_pred['AVG_LSTM_pred']
    df_wr = pd.merge(df_era, df_avg, how='inner', left_index=True, right_index=True,
                     on=['WLS_changed', 'NEXT_WLS',  'PE'])
    df_wr = df_wr[['WLS_changed', 'INN2_teampit', 'BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit',
                   'HR_pit', 'SB_pit', 'BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit',
                   'INN2_sp', 'ER_sp', 'AB_bat', 'RBI', 'RUN',
                   'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat', 'SB_bat', 'BB_bat', 'KK_bat',
                   'GD_bat', 'ERR', 'LOB', 'PE', '3hal', 'AVG_pred', 'ERA_pred', 'NEXT_WLS']]

    df_wr.to_csv('./data/result/for_wrate.csv', index=False)

def main():
    data = pd.read_csv('saber_plus_alpha.csv')
    X_era, y_era, X_avg, y_avg, era_columns, avg_columns = making_input(30, data)
    X_train_era, X_test_era, y_train_era, y_test_era = train_test_split(X_era, y_era, random_state=0, test_size=0.35)
    X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(X_avg, y_avg, random_state=0, test_size=0.35)

    X_train_std_era, X_test_std_era =  Min_Max_scaling_3D(X_train_era,  y_train_era, X_test_era,y_test_era)
    X_train_std_avg, X_test_std_avg = Min_Max_scaling_3D(X_train_avg, y_train_avg, X_test_avg, y_test_avg)
    train(X_train_std_era, y_train_era, 'era')
    train(X_train_std_avg, y_train_avg, 'avg')
    era_predict = predict('era', X_test_std_era)
    avg_predict = predict('avg', X_test_std_avg)
    for_lgb(X_test_era, y_test_era, era_predict.reshape(-1,), X_test_avg, y_test_avg, avg_predict.reshape(-1,), era_columns, avg_columns)


if __name__ == '__main__':
    main()

