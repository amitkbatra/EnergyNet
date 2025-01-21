import logging
import multiprocessing
import os
import pickle
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
import onnx
import onnxruntime as rt
import pandas as pd

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    filehandler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    return logger

def set_global_variables(num_trials, stats_sampling_rate, es_patience, es_restore_best_weights, batch_size, validation_split, epochs, devices, ml_task):
    global NUM_TRIALS, STATS_SAMPLING_RATE, ES_PATIENCE, ES_RESTORE_BEST_WEIGHTS, BATCH_SIZE, VALIDATION_SPLIT, EPOCHS, DEVICES, ML_TASK
    NUM_TRIALS = num_trials
    STATS_SAMPLING_RATE = stats_sampling_rate
    ES_PATIENCE = es_patience
    ES_RESTORE_BEST_WEIGHTS = es_restore_best_weights
    BATCH_SIZE = batch_size
    VALIDATION_SPLIT = validation_split
    EPOCHS = epochs
    DEVICES = devices
    ML_TASK = ml_task

# Socket for sending data from powerstat process to main process
HOST = "127.0.0.1"

# Define the measurement tool that will be used to gather power consumption data
power_consumption_measurement_tool = "powerstat"

# RAM
def get_ram_memory_uss(pid):
    process = psutil.Process(pid)
    return str(process.memory_full_info().uss / (1024*1024)) + ' MB'

def get_ram_memory_rss(pid):
    process = psutil.Process(pid)
    return str(process.memory_full_info().rss / (1024*1024)) + ' MB'

def get_ram_memory_vms(pid):
    process = psutil.Process(pid)
    return str(process.memory_full_info().vms / (1024*1024)) + ' MB'

def get_ram_memory_pss(pid):
    process = psutil.Process(pid)
    return str(process.memory_full_info().pss / (1024*1024)) + ' MB'

# CPU
def get_cpu_usage(pid):
    process = psutil.Process(pid)
    return str(process.cpu_percent(interval=0.5) / psutil.cpu_count()) + ' %'

def get_cpu_freq():
    return str(psutil.cpu_freq()[0]) + " MHz"

def perf(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    cmd = "echo pirata.lab | sudo -S -p \"\" perf stat -e power/energy-cores/,power/energy-pkg/"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    s.sendall(out.stdout)
    s.close()

def kill_perf():
    cmd = "echo pirata.lab | sudo -S -p \"\" pkill perf"
    subprocess.run(cmd, shell=True)

def powerstat(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    cmd = "echo pirata.lab | sudo -S -p \"\" powerstat -R 0.5 120"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    s.sendall(out.stdout)
    s.close()

def kill_powerstat():
    cmd = "echo pirata.lab | sudo -S -p \"\" pkill powerstat"
    subprocess.run(cmd, shell=True)

def get_cpu_power_draw():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    PORT = np.random.randint(10000, 20000)
    s.bind((HOST, PORT))
    s.listen()

    if power_consumption_measurement_tool == "powerstat":
        p = multiprocessing.Process(target=powerstat, args=(PORT,))
    elif power_consumption_measurement_tool == "perf":
        p = multiprocessing.Process(target=perf, args=(PORT,))

    p.start()
    conn, _addr = s.accept()
    time.sleep(max(1, STATS_SAMPLING_RATE / 2))

    if power_consumption_measurement_tool == "powerstat":
        q = multiprocessing.Process(target=kill_powerstat)
    elif power_consumption_measurement_tool == "perf":
        q = multiprocessing.Process(target=kill_perf)

    q.start()
    out = conn.recv(2048).decode()
    power_consumption = re.findall(r'CPU: (.+?) Watts', out)[0].strip() + " W"

    s.close()
    p.terminate()
    q.terminate()

    return power_consumption

# IO
def get_io_usage(pid):
    process = psutil.Process(pid)

    io_counters = process.io_counters()
    io_usage_process = io_counters[2] + io_counters[3] # read_bytes + write_bytes
    disk_io_counter = psutil.disk_io_counters()
    disk_io_total = disk_io_counter[2] + disk_io_counter[3] # read_bytes + write_bytes
    io_usage_process = io_usage_process / disk_io_total * 100
    io_usage_process = np.round(io_usage_process, 2)
    io_usage_process = str(io_usage_process) + " %"

    return io_usage_process

def get_bytes_written(pid):
    process = psutil.Process(pid)

    io_counters = process.io_counters()
    process_bytes_written = io_counters[3]
    total_bytes_written = psutil.disk_io_counters()[3]
    process_bytes_written = process_bytes_written / total_bytes_written * 100
    process_bytes_written = np.round(process_bytes_written, 2)
    process_bytes_written = str(process_bytes_written) + " %"

    return process_bytes_written

def get_bytes_read(pid):
    process = psutil.Process(pid)

    io_counters = process.io_counters()
    process_bytes_read = io_counters[2]
    total_bytes_read = psutil.disk_io_counters()[2]
    process_bytes_read = process_bytes_read / total_bytes_read * 100
    process_bytes_read = np.round(process_bytes_read, 2)
    process_bytes_read = str(process_bytes_read) + " %"

    return process_bytes_read

# GPU
get_gpu_memory_system = lambda: os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader').read().split('\n')[0] # SYSTEM WIDE

def get_gpu_memory(pid):
    output = os.popen('nvidia-smi | awk \'/' + str(pid) + '/{print $8}\'').read().split('\n')[0]
    output = "0 MiB" if output == "" else output.replace("MiB", "") + " MiB"

    return output

get_gpu_usage = lambda: os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader').read().split('\n')[0]
get_gpu_freq = lambda: os.popen('nvidia-smi --query-gpu=clocks.gr --format=csv,noheader').read().split('\n')[0]
get_gpu_power_draw = lambda: os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader').read().split('\n')[0]

def get_stats(pid):
    stats = {}
    stats['ram_memory_uss'] = get_ram_memory_uss(pid)
    stats['ram_memory_rss'] = get_ram_memory_rss(pid)
    stats['ram_memory_vms'] = get_ram_memory_vms(pid)
    stats['ram_memory_pss'] = get_ram_memory_pss(pid)
    stats['cpu_usage'] = get_cpu_usage(pid)
    stats['cpu_freq'] = get_cpu_freq()
    stats['cpu_power_draw'] = get_cpu_power_draw()
    stats['io_usage'] = get_io_usage(pid)
    stats['bytes_written'] = get_bytes_written(pid)
    stats['bytes_read'] = get_bytes_read(pid)
    stats['gpu_memory'] = get_gpu_memory(pid)
    stats['gpu_usage'] = get_gpu_usage()
    stats['gpu_freq'] = get_gpu_freq()
    stats['gpu_power_draw'] = get_gpu_power_draw()

    return stats

def sample_stats(test, sampling_rate, pid, directory):
    print(f"test: {test}")

    stats_list = []
    started = False

    Path(directory).mkdir(parents=True, exist_ok=True)

    while True:
        stats = get_stats(pid)
        stats_list.append(stats)

        # write stats to pickle file
        with open(f"{directory}/crypto_spider_5g_fcnn_optimized_benchmark_{test}_stats.pkl", 'wb') as f:
            pickle.dump(stats_list, f)

        if not started:
            # write file started.txt to signal that the sampling has started
            with open(f"started_{test}.txt", 'w') as f:
                f.write("STARTED")
            print("\nStats sampling started")

        started = True

        # check if file "stop.txt" exists
        if os.path.isfile(f"stop_{test}.txt"):
            print("Stats sampling stopped")
            os.remove(f"stop_{test}.txt")

            break
        else:
            time.sleep(sampling_rate)

def get_stats_background(test, sampling_rate, pid, directory):
    proc = Process(target=sample_stats, args=(test, sampling_rate, pid, directory))
    proc.start()

    return proc

def strip_units(x):
    return float(x.split(' ')[0])

def agg_stats(agg_func, stats_list, average_time_spent):
    average_stats = stats_list.copy()

    # strip units of the stats of every trial in stats_list
    for trial in average_stats:
        for snapshot in trial:
            for stat in snapshot:
                stats_value = snapshot[stat]
                stats_value_stripped = strip_units(stats_value)
                snapshot[stat] = stats_value_stripped

    trials_list = []
    
    # convert to a numpy array
    for trial in average_stats:
        df = pd.DataFrame(trial)
        trial = df.to_numpy()
        trials_list.append(trial)
    
    trials_list_np = np.array(trials_list)
    
    print("trials_list_np.shape: {}".format(trials_list_np.shape))
    
    # fill first axis of trials_list_np with NaNs until all trials have the same length
    max_length = max([trial.shape[0] for trial in trials_list_np])
    trials_list_np_filled = []
    
    for trial in trials_list_np:
        trial_length = trial.shape[0]
        
        if trial_length < max_length:
            print(f"Trial length ({trial_length}) is smaller than max length ({max_length}). Filling with NaNs...")
        
            # fill first axis of trial with NaNs until trial has the same length as the longest trial
            trial = np.pad(trial, ((0, max_length - trial_length), (0, 0)), 'constant', constant_values=np.nan)
            
            print("trial.shape: {}".format(trial.shape))
        
        trials_list_np_filled.append(trial)
    
    trials_list_np_filled = np.array(trials_list_np_filled)
    
    print("trials_list_np_filled.shape: {}".format(trials_list_np_filled.shape))

    average_stats_np = agg_func(trials_list_np_filled, axis=0)

    print("average_stats_np.shape: {}".format(average_stats_np.shape))

    return average_stats_np

def get_average_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanmean, stats_list=stats_list, average_time_spent=average_time_spent)

def get_std_dev_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanstd, stats_list=stats_list, average_time_spent=average_time_spent)

def get_max_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanmax, stats_list=stats_list, average_time_spent=average_time_spent)

def save_stats_to_logfile(test, average_stats, std_dev_stats, max_stats, logger):
    units = {'ram_memory_uss': 'MB', 'ram_memory_rss': 'MB', 'ram_memory_vms': 'MB', 'ram_memory_pss': 'MB', 'cpu_usage': '%', 'cpu_freq': 'MHz', 'cpu_power_draw': 'W', 'io_usage': '%', 'bytes_written': 'MB', 'bytes_read': 'MB', 'gpu_memory': 'MB', 'gpu_usage': '%', 'gpu_freq': 'MHz', 'gpu_power_draw': 'W'}

    for key in average_stats.keys():
        logger.info(f'[{test}] {key} (average): {average_stats[key]} {units[key]}')
        logger.info(f'[{test}] {key} (std_dev): {std_dev_stats[key]} {units[key]}')
        logger.info(f'[{test}] {key} (max): {max_stats[key]} {units[key]}')

@contextmanager
def measure_time() -> float:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

class StatsCollectionManager():
    def __init__(self, test, sampling_rate=0.1, pid=None, directory=None):
        self.test = test
        self.sampling_rate = sampling_rate
        self.pid = pid
        self.proc = None
        
        if directory is None:
            self.directory = f"poc_energy_efficiency_crypto/"
        else:
            self.directory = f"poc_energy_efficiency_crypto/{directory}/"

        print("Starting stats collection for test: {}".format(test))

    def __enter__(self):
        self.proc = get_stats_background(test=self.test, sampling_rate=self.sampling_rate, pid=self.pid, directory=self.directory)

        while True:
            # check if file stats.txt exists
            if os.path.exists(f"started_{self.test}.txt"):
                print("\nStats collection started")
                os.remove(f"started_{self.test}.txt")
                break

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file stop.txt to signal to the background process to stop
        with open(f"stop_{self.test}.txt", "w") as f:
            print("Stopping stats collection")
            f.write("STOP")

        while True:
            if os.path.isfile(f"{self.directory}/crypto_spider_5g_fcnn_optimized_benchmark_{self.test}_stats.pkl"):
                print("Stats file found")
                print(self.test)

                break

def perform_inference_tflite(interpreter, batch_size, data_transformed):
    input_details = interpreter.get_input_details()[0]
    dtype = input_details['dtype']
    input_data = np.array(data_transformed, dtype=dtype)

    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []

    for i in range(0, len(data_transformed), batch_size):
        batch = data_transformed[i:i+batch_size]
        batch_data = np.array(batch, dtype=dtype)

        if len(batch) == batch_size:
            interpreter.set_tensor(input_details['index'], batch_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            preds.append(output_data)

    return np.concatenate(preds)

def save_model_to_tflite(tflite_model, model_filename):
    with open(f"{model_filename}.tflite", 'wb') as f:
        f.write(tflite_model)
    return f"{model_filename}.tflite"

def gzip_model(model_path):
    with open(model_path, 'rb') as f_in, open(f"{model_path}.gz", 'wb') as f_out:
        f_out.write(gzip.compress(f_in.read()))
    os.remove(model_path)
    return f"{model_path}.gz"

def unzip_model(model_path):
    with gzip.open(model_path, 'rb') as f_in, open(model_path[:-3], 'wb') as f_out:
        f_out.write(f_in.read())
    return model_path[:-3]

def load_model_from_tflite(model_path) -> tf.lite.Interpreter:
    tflite_model = tf.lite.Interpreter(model_path=model_path)
    tflite_model.allocate_tensors()
    return tflite_model

def load_model(ext):
    model_path = f"model.{ext}.gz"
    model_path = unzip_model(model_path)

    if ext == "tflite":
        model = load_model_from_tflite(model_path)
    else:
        raise Exception("Model format not supported")

    return model

def get_gzipped_model_size(ext):
    if os.path.exists(f"model.{ext}.gz"):
        return os.path.getsize(f"model.{ext}.gz")
    else:
        RuntimeError(f"Model with extension \"{ext}\" not found")

def delete_model(ext):
    if os.path.exists(f"model.{ext}.gz"):
        os.remove(f"model.{ext}.gz")
    else:
        RuntimeError(f"Model with extension \"{ext}\" not found")

    if os.path.exists(f"model.{ext}"):
        os.remove(f"model.{ext}")
    else:
        RuntimeError(f"Model with extension \"{ext}\" not found")

def perform_evaluation_tflite(experiment, device, df_test, standard, logger):
    assert device in ["CPU", "GPU"]

    logger.info(f"Evaluating {experiment} model on test set")

    # Load model
    interpreter = load_model(ext="tflite")
    interpreter.allocate_tensors()

    # Load test set
    test_data_transformed = standard.transform(df_test[gf])

    # get input shape
    input_shape = interpreter.get_input_details()[0]["shape"]
    logger.info(f"Input shape: {input_shape}")

    # get output shape
    output_shape = interpreter.get_output_details()[0]["shape"]
    logger.info(f"Output shape: {output_shape}")

    # transform data to the expected tensor type
    input_details = interpreter.get_input_details()[0]
    dtype = input_details["dtype"]
    input_data = np.array(test_data_transformed, dtype=dtype)

    # reshape model input
    batch_size = 256

    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []

    # create batches of test_data_transformed
    for i in range(0, len(test_data_transformed), batch_size):
        batch = test_data_transformed[i:i+batch_size]
        batch_data = np.array(batch, dtype=dtype)

        if len(batch) == batch_size:
            interpreter.set_tensor(input_details['index'], batch_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            preds.append(output_data)

    predictions = np.concatenate(preds)

    # take the labels of the test set
    test_labels = df_test["tag"].values[:len(predictions)]

    if ML_TASK == "binary_classification":
        # convert predictions to labels
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted")
        auc = roc_auc_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions, average="weighted")
        precision = precision_score(test_labels, predictions, average="weighted")
        balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
        matthews = matthews_corrcoef(test_labels, predictions)

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 score: {f1}")
        logger.info(f"AUC: {auc}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Balanced accuracy: {balanced_accuracy}")
        logger.info(f"Matthews correlation coefficient: {matthews}")

        test_results = {
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
            "recall": recall,
            "precision": precision,
            "balanced_accuracy": balanced_accuracy,
            "matthews": matthews
        }

        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, accuracy, f1, auc, recall, precision, balanced_accuracy, matthews]], columns=["experiment", "device", "accuracy", "f1", "auc", "recall", "precision", "balanced_accuracy", "matthews"])
    elif ML_TASK == "regression":
        mae = mean_absolute_error(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        mape = mean_absolute_percentage_error(test_labels, predictions)
        smape = 1/len(test_labels) * np.sum(2 * np.abs(predictions - test_labels) / (np.abs(predictions) + np.abs(test_labels)))

        logger.info(f"MAE: {mae}")
        logger.info(f"MSE: {mse}")
        logger.info(f"MAPE: {mape}")
        logger.info(f"SMAPE: {smape}")

        test_results = {
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "smape": smape
        }

        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, mae, mse, mape, smape]], columns=["experiment", "device", "mae", "mse", "mape", "smape"])

    with open("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.pkl", "wb") as f:
        pickle.dump(test_results, f)

    return df_evaluation

def benchmark_training(device, model, history, model_path):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    pid = os.getpid()

    with StatsCollectionManager(test="training", sampling_rate=STATS_SAMPLING_RATE, pid=pid) as training_scm:
        with measure_time() as training_time_measure:
            with tf.device(device):
                pass

    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    # Get training time
    training_time = training_time_measure()

    # Write training time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_time.pkl", "wb") as f:
        pickle.dump({"training_time": training_time}, f)

def benchmark_inference(device, data_transformed):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    pid = os.getpid()

    # load model
    interpreter = load_model(ext="tflite")

    # Inference test
    with StatsCollectionManager(test="inference", sampling_rate=STATS_SAMPLING_RATE, pid=pid) as inference_scm:
        with measure_time() as inference_time_measure:
            with tf.device(device):
                perform_inference_tflite(interpreter, batch_size=BATCH_SIZE, data_transformed=data_transformed)

    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    # Get inference and load times
    inference_time = inference_time_measure()

    # Write inference time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_time.pkl", "wb") as f:
        pickle.dump({"inference_time": inference_time}, f)

def benchmark_load(device):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    pid = os.getpid()

    # load test
    with StatsCollectionManager(test="load", sampling_rate=STATS_SAMPLING_RATE, pid=pid) as load_scm:
        with measure_time() as load_time_measure:
            with tf.device(device):
                load_model(ext="tflite")

    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    # Get load time
    load_time = load_time_measure()

    # Write load time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_time.pkl", "wb") as f:
        pickle.dump({"load_time": load_time}, f)
