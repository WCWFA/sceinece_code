import argparse
import csv
import os
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--data', type=str, default='household_power_consumption.csv',
                        help="Path to the dataset")
    parser.add_argument('--time_steps', type=int, default=36)
    return parser.parse_args()

def load_data(data_name):
    path = os.path.join('data', data_name)    #修改文件夹路径
    dataset = pd.read_csv(path)
    dataset = dataset.replace('?', np.nan)  #将数据集中的？替换为NAN
    dataset.iloc[:, 2:] = dataset.iloc[:, 2:].astype(float)  #将数据集中第三列开始的浮点数列的数据类型转换为浮点型。
    dataset = dataset[(dataset.iloc[:, 2:] != 0).all(axis=1) & dataset.notna().all(axis=1)]    #删除所有数据集中存在 0 或 NaN 的行。
    # print(dataset)
    data = dataset.values   #将数据集转换为 Numpy 数组。
    # encoder = OneHotEncoder()
    # one_hot = encoder.fit_transform(np.expand_dims(data[:, -4], axis=1)).toarray()
    # data = np.delete(data, -4, axis=1)
    # data = np.concatenate((data, one_hot), axis=1)
    # data = data.astype('float32')
    date_column = data[:, 0]
    time_column = data[:, 1]  #将日期和时间列分别存储
    float_times = []
    float_dates = []
    for date_str, time_str in zip(date_column, time_column):  #循环迭代日期列和时间列中的元素：
        time = datetime.datetime.strptime(time_str, "%H:%M:%S")
        date = datetime.datetime.strptime(date_str, "%d/%m/%Y")   #将时间和日期字符串按给定格式解析为datetime 对象。
        float_time = (time.hour * 3600 + time.minute * 60 + time.second) / 86400.0   #将时间转换为浮点数，表示一天中的比例。
        float_date = date.timestamp()    #将日期转换为浮点数，表示自 1970 年 1 月 1 日以来的秒数。
        float_times.append(float_time)
        float_dates.append(float_date)    #将转换后的时间和日期存储在两个列表中
    data[:, 0] = np.array(float_dates)
    data[:, 1] = np.array(float_times)    #将转换后的浮点数时间和日期分别存储回数据数组中的第一列和第二列。
    data = data.astype('float32')
    # print(data)
    real_y = data[:, 2]
    data = np.delete(data, 2, axis=1)
    real_y = real_y.reshape(-1, 1)   #重塑目标变量的形状为一列。
    data = np.hstack((real_y, data))
    # print(data)
    scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(data[:, 1:])
    scaled_y = y_scaler.fit_transform(np.expand_dims(data[:, 0], axis=1))
    scaled_data = np.concatenate((scaled_y, scaled_X), axis=1)
    return scaled_data, y_scaler

def data_to_series_features(data, time_steps):   #用于将时间序列数据转换为序列特征和目标值的函数。
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

    args.n_features = np.size(data, axis=-2)
    X, y = data_to_series_features(data, args.time_steps)

    train_X, X, train_y, y = train_test_split(X, y, test_size=0.2, random_state=28333)
    valid_X, test_X, valid_y, test_y = train_test_split(X, y, test_size=0.5, random_state=28333)

    gbrt_model = GradientBoostingRegressor()
    gbrt_model.fit(train_X, train_y.ravel())

    # 验证集上的预测
    valid_predictions = gbrt_model.predict(valid_X)
    valid_predictions = y_scaler.inverse_transform(valid_predictions.reshape(-1, 1))
    valid_y_transform = y_scaler.inverse_transform(valid_y.reshape(-1, 1))

    # 计算并打印RMSE和MAE
    rmse_valid = np.sqrt(mean_squared_error(valid_y_transform, valid_predictions))
    mae_valid = mean_absolute_error(valid_y_transform, valid_predictions)
    r2_valid = r2_score(valid_y_transform, valid_predictions)
    mape_valid = calculate_mape(valid_y_transform, valid_predictions)

    print("Validation MAPE:", mape_valid)
    print("Validation R2:", r2_valid)
    print("Validation RMSE:", rmse_valid)
    print("Validation MAE:", mae_valid)


    # 测试集上的预测
    test_predictions = gbrt_model.predict(test_X)
    test_predictions = y_scaler.inverse_transform(test_predictions.reshape(-1, 1))
    test_y_transform = y_scaler.inverse_transform(test_y.reshape(-1, 1))

    # 计算并打印测试集上的RMSE和MAE
    rmse_test = np.sqrt(mean_squared_error(test_y_transform, test_predictions))
    mae_test = mean_absolute_error(test_y_transform, test_predictions)
    r2_test = r2_score(test_y_transform, test_predictions)
    mape_test = calculate_mape(test_y_transform, test_predictions)

    print("Test MAPE:", mape_test)
    print("Test R2:", r2_test)
    print("Test RMSE:", rmse_test)
    print("Test MAE:", mae_test)

    def save_to_csv(filename, index, true_y, predicted_y):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['ORDER', 'true_y', 'predicted_y']
            csvwriter.writerow(header)
            for i, (idx, true, predicted) in enumerate(zip(index, true_y, predicted_y)):
                csvwriter.writerow([idx, true[0], predicted[0]])  # 注意反归一化后的值需要提取 [0]

    # 保存反归一化后的数据到CSV文件
    save_to_csv('gbrt_valid_data.csv', valid_data['ORDER'], valid_y_transform, valid_predictions)
    save_to_csv('gbrt_test_data.csv', test_data['ORDER'], test_y_transform, test_predictions)


main()