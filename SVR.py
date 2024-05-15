import csv
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
from sklearn.svm import SVR
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--data', type=str, default='household_power_consumption.csv',
                        help="Path to the dataset")
    parser.add_argument('--time_steps', type=int, default=36)
    return parser.parse_args()

def load_data(data_name):
    path = os.path.join('data', data_name)
    dataset = pd.read_csv(path)
    dataset = dataset.replace('?', np.nan)
    dataset.iloc[:, 2:] = dataset.iloc[:, 2:].astype(float)
    dataset = dataset[(dataset.iloc[:, 2:] != 0).all(axis=1) & dataset.notna().all(axis=1)]
    print(dataset)
    data = dataset.values
    # encoder = OneHotEncoder()
    # one_hot = encoder.fit_transform(np.expand_dims(data[:, -4], axis=1)).toarray()
    # data = np.delete(data, -4, axis=1)
    # data = np.concatenate((data, one_hot), axis=1)
    # data = data.astype('float32')
    date_column = data[:, 0]
    time_column = data[:, 1]
    float_times = []
    float_dates = []
    for date_str, time_str in zip(date_column, time_column):
        time = datetime.datetime.strptime(time_str, "%H:%M:%S")
        date = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        float_time = (time.hour * 3600 + time.minute * 60 + time.second) / 86400.0
        float_date = date.timestamp()
        float_times.append(float_time)
        float_dates.append(float_date)
    data[:, 0] = np.array(float_dates)
    data[:, 1] = np.array(float_times)
    data = data.astype('float32')
    print(data)
    real_y = data[:, 2]
    data = np.delete(data, 2, axis=1)
    real_y = real_y.reshape(-1, 1)
    data = np.hstack((real_y, data))
    print(data)
    scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(data[:, 1:])
    scaled_y = y_scaler.fit_transform(np.expand_dims(data[:, 0], axis=1))
    scaled_data = np.concatenate((scaled_y, scaled_X), axis=1)
    return scaled_data, y_scaler

def data_to_series_features(data, time_steps):     #将数据转换为时间序列特征函数
    data_size = len(data) - time_steps
    series_X = []
    series_y = []
    for i in range(data_size):
        series_X.append(data[i:i + time_steps])
        series_y.append(data[i + time_steps, 0])
    series_X = np.array(series_X)
    series_y = np.array(series_y)
    # 将 series_X 转换为二维数组
    series_X = series_X.reshape(series_X.shape[0], -1)
    return series_X, series_y

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    args = parse_arguments()
    data, y_scaler = load_data(args.data)
    args.n_features = np.size(data, axis=-1)
    X, y = data_to_series_features(data, args.time_steps)
    train_X, X, train_y, y = train_test_split(X, y, test_size=0.2, random_state=28333)
    valid_X, test_X, valid_y, test_y = train_test_split(X, y, test_size=0.5, random_state=28333)
    print("train_X shape:", train_X.shape)
    svr_model = SVR(kernel='poly', degree=3)
    # # 使用tqdm显示模型拟合进度
    # with tqdm(total=len(train_X), desc="Fitting SVR model") as pbar:
    #     for _ in svr_model.fit(train_X, train_y):
    #         pbar.update(1)
    svr_model.fit(train_X, train_y)


    # predictions = svr_model.predict(valid_X)
    # predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1))
    # # valid_y = y_scaler.inverse_transform(valid_y.reshape(-1, 1))
    # #
    # # rmse = np.sqrt(mean_squared_error(valid_y, predictions))
    # # mae = mean_absolute_error(valid_y, predictions)
    # #
    # # print("RMSE:", rmse)
    # # print("MAE:", mae)
    # test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    # rmse = np.sqrt(mean_squared_error(test_y, predictions))
    # mae = mean_absolute_error(test_y, predictions)
    # print("RMSE:", rmse)
    # print("MAE:", mae)
    # 验证集上的预测
    valid_predictions = svr_model.predict(valid_X)
    valid_predictions = y_scaler.inverse_transform(valid_predictions.reshape(-1, 1))
    valid_y_transform = y_scaler.inverse_transform(valid_y.reshape(-1, 1))

    # 计算并打印RMSE和MAE
    rmse_valid = np.sqrt(mean_squared_error(valid_y_transform, valid_predictions))
    mae_valid = mean_absolute_error(valid_y_transform, valid_predictions)
    r2_valid  = r2_score(valid_y_transform, valid_predictions)
    mape_valid  = calculate_mape(valid_y_transform, valid_predictions)

    print("Validation MAPE:", mape_valid)
    print("Validation R2:", r2_valid)
    print("Validation RMSE:", rmse_valid)
    print("Validation MAE:", mae_valid)

    # 测试集上的预测
    test_predictions = svr_model.predict(test_X)
    test_predictions = y_scaler.inverse_transform(test_predictions.reshape(-1, 1))
    test_y_transform = y_scaler.inverse_transform(test_y.reshape(-1, 1))

    # 计算并打印测试集上的RMSE和MAE
    rmse_test = np.sqrt(mean_squared_error(test_y_transform, test_predictions))
    mae_test = mean_absolute_error(test_y_transform, test_predictions)
    r2_test  = r2_score(test_y_transform, test_predictions)
    mape_test  = calculate_mape(test_y_transform, test_predictions)

    print("Test MAPE:", mape_test)
    print("Test R2:", r2_test)
    print("Test RMSE:", rmse_test)
    print("Test MAE:", mae_test)

    # 保存训练集、验证集和测试集的真实值和预测值到CSV文件
    def save_to_csv(filename, true_y, predicted_y):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['true_y', 'predicted_y']
            csvwriter.writerow(header)
            for true, predicted in zip(true_y, predicted_y):
                csvwriter.writerow([true[0], predicted[0]])  # 注意反归一化后的值需要提取 [0]

    # 保存反归一化后的数据到CSV文件
    save_to_csv('svr_valid_data.csv', valid_y_transform, valid_predictions)
    save_to_csv('svr_test_data.csv', test_y_transform, test_predictions)

main()