import pickle
import sys
import signal
from matplotlib import pyplot as plt
import csv
from utils import (load_data, data_to_series_features,
                   apply_weight, is_minimum)
from algorithm import (initialize_weights, individual_to_key,
                       pop_to_weights, select, reconstruct_population)
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
from keras import optimizers
from keras.models import clone_model
import tensorflow as tf
import argparse
import math
import numpy as np
from model import make_model
from copy import copy
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import EarlyStopping

# 检查GPU数量
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 如果有两个或更多GPU可用，使用两个GPU
    if len(gpus) >= 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用两个GPU，编号为0和1
        print('现在正在使用两张GPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用单个GPU，编号为0
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print('没有GPU可用，将使用CPU')

# 指定保存模型权重的文件名
model_weights_file = 'model_weights.h5'
progress_state_file = 'progress_state.pkl'

def save_model_weights(model, filename):
    model.save_weights(filename)

def load_model_weights(model, filename):
    model.load_weights(filename)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定要使用的 GPU 设备编号
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # 仅使用第一个 GPU 设备
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

device_name = tf.test.gpu_device_name()
if device_name != '':
    print('GPU')
else:
    print('CPU')


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--iterations', type=int, default=3,     #原数是20
                        help="Specify the number of evolution iterations")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")
    parser.add_argument('--initial_epochs', type=int, default=100,     #原数是100
                        help="Specify the number of epochs for initial training")
    parser.add_argument('--num_epochs', type=int, default=4,      #原数是40
                        help="Specify the number of epochs for competitive search")
    parser.add_argument('--log_step', type=int, default=100,
                        help="Specify log step size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument('--data', type=str, default='household_power_consumption.csv',
                        help="Path to the dataset")
    parser.add_argument('--pop_size', type=int, default=6)     #原数是36
    parser.add_argument('--code_length', type=int, default=6)
    parser.add_argument('--n_select', type=int, default=6)
    parser.add_argument('--time_steps', type=int, default=36)
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    return parser.parse_args()


class CrashCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path):
        super(CrashCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model.save_weights(self.checkpoint_path, overwrite=True)
        except Exception as e:
            print("Exception occurred during model checkpointing:", str(e))
            raise

def load_state(filename):
    try:
        with open(filename, 'rb') as file:
            state = pickle.load(file)
        return state
    except FileNotFoundError:
        return None
    except Exception as e:
        print("加载状态时出错：", str(e))
        return None

def save_state(filename, state):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(state, file)
    except Exception as e:
        print("保存状态时出错：", str(e))

def save_progress_and_exit(signal, frame):
    global progress_state
    print("Received Ctrl+C. Saving progress...")
    save_state(progress_state_file, progress_state)
    sys.exit(1)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    args = parse_arguments()
    # 加载之前保存的进度状态
    progress_state = load_state(progress_state_file)
    if progress_state is None:
        progress_state = {'iteration': 0, 'enum': 0}
    else:
        print("Resuming from progress state:", progress_state)

    # 添加信号处理程序来捕获Ctrl+C
    signal.signal(signal.SIGINT, save_progress_and_exit)
    data, y_scaler = load_data(args.data)
    args.n_features = np.size(data, axis=-1)
    X, y = data_to_series_features(data, args.time_steps)
    train_X, X, train_y, y = train_test_split(X, y, test_size=0.2, random_state=28333)
    valid_X, test_X, valid_y, test_y = train_test_split(X, y, test_size=0.5, random_state=28333)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate, clipnorm=args.max_grad_norm)
    # optimizer = optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.max_grad_norm)
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  # 将模型放置在GPU设备编号为0的GPU上
    best_model = make_model(args)
    best_weight = [1.0] * args.time_steps
    best_model.compile(loss='mse', optimizer=optimizer)
    # 检查是否存在已保存的模型权重文件，如果存在，则加载它们
    if os.path.exists(model_weights_file):
        load_model_weights(best_model, model_weights_file)
        print("Loaded model weights from:", model_weights_file)

    # # 检查是否存在先前保存的检查点，如果存在，则加载检查点
    # checkpoint_file = os.path.join(checkpoint_dir, 'model_checkpoint.h5')
    # if os.path.isfile(checkpoint_file):
    #     best_model.load_weights(checkpoint_file)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if progress_state['iteration'] == 0:
        print("Initial training before competitive random search")
        # best_model.fit(apply_weight(train_X, best_weight), train_y, epochs=args.initial_epochs,
        #                validation_data=(apply_weight(valid_X, best_weight), valid_y), shuffle=True,
        #                callbacks=[early_stopping])
        # 记录每个训练周期的val_loss值
        history = best_model.fit(apply_weight(train_X, best_weight), train_y, epochs=args.initial_epochs,
                                 validation_data=(apply_weight(valid_X, best_weight), valid_y), shuffle=True)

        # 提取val_loss值和训练周期数
        val_loss = history.history['val_loss']
        epochs = range(1, args.initial_epochs + 1)
        loss = history.history['loss']

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_loss, 'b', label='Validation')
        plt.plot(epochs, loss, 'r', label='Train')
        plt.title('Initial training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_validation_loss.png')


        # 绘制折线图
        plt.figure(1)
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Initial training')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        # 保存图像文件
        plt.savefig('validation_loss_vs_epochs.png')

        # 绘制loss折线图
        plt.figure(2)
        plt.plot(epochs, loss, 'b', label='Loss')
        plt.title('Initial training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # 保存图像
        plt.savefig("loss_vs_epochs.png")

        print("\nInitial training is done. Start competitive random search.\n")
    pop, weights = initialize_weights(args.pop_size, args.time_steps, args.code_length)
    key_to_rmse = {}
    # crash_checkpoint = CrashCheckpoint(checkpoint_file)

    for iteration in range(progress_state['iteration'], args.iterations):
        for enum, (indiv, weight) in enumerate(zip(pop[progress_state['enum']:], weights[progress_state['enum']:])):
            try:
                print(
                    'iteration: [%d/%d] indiv_no: [%d/%d]' % (iteration + 1, args.iterations, enum + 1, args.pop_size))
                key = individual_to_key(indiv)
                if key not in key_to_rmse.keys():
                    model = make_model(args)
                    model.compile(loss='mse', optimizer=optimizer)
                    model.set_weights(best_model.get_weights())
                    model.fit(apply_weight(train_X, weight), train_y, epochs=args.num_epochs,
                              validation_data=(apply_weight(valid_X, weight), valid_y),
                              shuffle=True)
                    pred_y = model.predict(apply_weight(valid_X, weight))
                    inv_pred_y = y_scaler.inverse_transform(pred_y)
                    inv_valid_y = y_scaler.inverse_transform(np.expand_dims(valid_y, axis=1))
                    rmse = math.sqrt(mean_squared_error(inv_valid_y, inv_pred_y))
                    mae = mean_absolute_error(inv_valid_y, inv_pred_y)
                    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))
                    if is_minimum(rmse, key_to_rmse):
                        best_model.set_weights(model.get_weights())
                        best_weight = copy(weight)
                    key_to_rmse[key] = rmse
            except Exception as e:
                print("Exception occurred:", str(e))
                print("Saving progress...")
                progress_state['iteration'] = iteration
                progress_state['enum'] = enum
                # 保存模型权重和进程状态
                save_state(progress_state_file, progress_state)
                save_model_weights(best_model, model_weights_file)
                raise
        pop_selected, fitness_selected = select(pop, args.n_select, key_to_rmse)
        pop = reconstruct_population(pop_selected, args.pop_size)
        weights = pop_to_weights(pop, args.time_steps, args.code_length)

        # 在每个训练迭代结束后保存模型权重
        save_model_weights(best_model, model_weights_file)

        # 保存进度状态
        progress_state['iteration'] = iteration + 1
        progress_state['enum'] = 0
        save_state(progress_state_file, progress_state)

    print('valid evaluation:')
    pred_y_valid = best_model.predict(apply_weight(valid_X, best_weight))
    valid_pred_y = y_scaler.inverse_transform(pred_y_valid)
    inv_valid_y = y_scaler.inverse_transform(np.expand_dims(valid_y, axis=1))

    rmse = math.sqrt(mean_squared_error(inv_valid_y, valid_pred_y))
    mae = mean_absolute_error(inv_valid_y, valid_pred_y)
    r2 = r2_score(inv_valid_y, valid_pred_y)
    mape = calculate_mape(inv_valid_y, valid_pred_y)
    print("MAPE:", mape)
    print("R2:", r2)
    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))



    print('test evaluation:')
    pred_y = best_model.predict(apply_weight(test_X, best_weight))
    inv_pred_y = y_scaler.inverse_transform(pred_y)
    inv_test_y = y_scaler.inverse_transform(np.expand_dims(test_y, axis=1))

    rmse = math.sqrt(mean_squared_error(inv_test_y, inv_pred_y))
    mae = mean_absolute_error(inv_test_y, inv_pred_y)
    r2 = r2_score(inv_test_y, inv_pred_y)
    mape = calculate_mape(inv_test_y, inv_pred_y)
    print("MAPE:", mape)
    print("R2:", r2)
    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))

    # 保存训练集、验证集和测试集的真实值和预测值到CSV文件
    def save_to_csv(filename, true_y, predicted_y):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['true_y', 'predicted_y']
            csvwriter.writerow(header)
            for true, predicted in zip(true_y, predicted_y):
                csvwriter.writerow([true[0], predicted[0]])  # 注意反归一化后的值需要提取 [0]

    # 保存反归一化后的数据到CSV文件
    save_to_csv('EA-LSTM_valid_data.csv', inv_valid_y, valid_pred_y)
    save_to_csv('EA-LSTM_test_data.csv', inv_test_y, inv_pred_y)

    # 打印模型的摘要信息
    best_model.summary()

    # 绘制模型结构图
    tf.keras.utils.plot_model(best_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



if __name__ == '__main__':
    main()
