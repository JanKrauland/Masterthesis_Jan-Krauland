import ast
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_dir")
args = parser.parse_args()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

metrics_file = os.path.join(args.metrics_dir, 'metrics.json')

with open(metrics_file, 'r') as f:
    metrics = [ast.literal_eval(l[:-1]) for l in f.readlines()]
    f.close()

try:
    # Filtern Sie die Daten basierend auf vorhandenen Iterationen und Werten
    filtered_loss = [m for m in metrics if 'iteration' in m and 'total_loss' in m and 'total_val_loss' in m]
    filtered_ap = [m for m in metrics if 'iteration' in m and 'bbox/AP50' in m]
    filtered_lr = [m for m in metrics if 'iteration' in m and 'lr' in m]
# Weiterer Code für die Plot-Erstellung hier

    train_loss = [float(v['total_loss']) for v in filtered_loss]
    val_loss = [float(v['total_val_loss']) for v in filtered_loss]
    map = [float(v['bbox/AP50']) for v in filtered_ap]
    lr = [float(v['lr']) for v in filtered_lr]
    # Erstellen Sie die x-Werte basierend auf den Iterationen
    x_values_loss = [m['iteration'] for m in filtered_loss]
    x_values_map = [m['iteration'] for m in filtered_ap]
    x_values_lr = [m['iteration'] for m in filtered_lr]

    window_size = 5
    train_loss_avg = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
    val_loss_avg = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')

    #train_loss_avg = moving_average(train_loss, n=30)
    #val_loss_avg = moving_average(val_loss, n=30)
    #map_average = moving_average(map, n=10)
    x_values_loss = x_values_loss[:len(train_loss_avg)]
    #x_map = x_map[:len(map_average)]
    
    # Plot loss
    plt.plot(x_values_loss, train_loss_avg, label='train loss')
    plt.plot(x_values_loss, val_loss_avg, label='val loss')
    plt.xlabel('Iterationen')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(metrics_file), "Train_Val_loss"))
    
    # Plot mAP
    plt.figure()
    plt.plot(x_values_map, map, label='mAP@50')
    plt.xlabel('Iterationen')
    plt.ylabel('mAP@50')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(metrics_file), "mAP@50"))

    # Plot LR
    plt.figure()
    plt.plot(x_values_lr, lr, label='Learning Rate')
    plt.xlabel('Iterationen')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(metrics_file), "Learning Rate"))
except Exception as e:
    print(e)