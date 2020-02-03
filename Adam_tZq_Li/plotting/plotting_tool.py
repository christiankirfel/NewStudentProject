import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
from mpl_toolkits import mplot3d

label_changing_parameter_1='Lr'
label_changing_parameter_2= ''
mode="2d"
directory = '/cephfs/user/s6ribaum/python_submissions/test/output/3j1b_final/Lr/'
output_path = '/cephfs/user/s6ribaum/python_submissions/test/plotting/' + label_changing_parameter_1 + '_' + label_changing_parameter_2 + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#Reads out data from all subdir with the ending txt. Then comprehends them in a pd file, for further plotting.
def readout(directory):
    first_entry = True
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".txt"):
                #print (filepath)
                if (first_entry == True):
                    df = pandas.read_csv(filepath, delimiter = ";", header=None, names=["Adam", "Lr", "Nodes", "Layers", "Dropout", "Validation", "Batchsize", "Decay", "Momentum", "Epoch", "2j1b", "accuracy", "loss", "val_accuracy", "val_loss"])
                    first_entry = False
                else :
                    df_temp = pandas.read_csv(filepath, delimiter=";", header=None, names=["Adam", "Lr", "Nodes", "Layers", "Dropout", "Validation", "Batchsize", "Decay", "Momentum", "Epoch", "2j1b", "accuracy", "loss", "val_accuracy", "val_loss"])
                    df = df.append(df_temp, ignore_index=True)
    return df

def plot_loss_2d(dataframe, label_changing_parameter):
    plt.plot(dataframe[label_changing_parameter], dataframe['loss'], 'b+' ,label='$ Loss_{Training} $')
    plt.plot(dataframe[label_changing_parameter], dataframe['val_loss'], 'r+', label='$ Loss_{Validation} $')
    title_labes = ''
    for i in range(1, 8): 
        if (label_changing_parameter != dataframe.columns[i]):
            title_labes = title_labes + dataframe.columns[i] + '=' + str(dataframe[dataframe.columns[i]][0]) 
            if (i != 7):
                title_labes = title_labes + ', '
    plt.title(title_labes, y=1.05, fontsize=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel('Loss')
    #plt.xlim(1E-1, 1E-9)
    plt.ylim(0.0005, 0.0012)
    plt.xscale("log")
    plt.xlabel(label_changing_parameter)
    plt.legend(loc="upper right")
    plt.gcf().savefig(output_path + 'losses.png', dpi=400)
    plt.gcf().clear()

def plot_accuracy_2d(dataframe, label_changing_parameter):
    plt.plot(dataframe[label_changing_parameter], dataframe['accuracy'], 'b+', label='$ Accuracy_{Training} $')
    plt.plot(dataframe[label_changing_parameter], dataframe['val_accuracy'], 'r+', label='$ Accuracy_{Validation} $')
    title_labes = ''
    for i in range(1, 8): 
        if (label_changing_parameter != dataframe.columns[i]):
            title_labes = title_labes + dataframe.columns[i] + '=' + str(dataframe[dataframe.columns[i]][0]) 
            if (i != 7):
                title_labes = title_labes + ', '
    plt.title(title_labes, y=1.05, fontsize=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel('Loss')
    plt.ylabel('accuracy')
    plt.xscale("log")
    plt.xlabel(label_changing_parameter)
    plt.legend(loc="upper left")
    plt.gcf().savefig(output_path + 'accuracy.png', dpi=300)
    plt.gcf().clear()

def plot_3d(dataframe, label_metric, metric, label_changing_parameter_1, label_changing_parameter_2):
    title_labes = ''
    for i in range(1, 8): 
        if (label_changing_parameter_1 != dataframe.columns[i] and label_changing_parameter_2 != dataframe.columns[i]):
            title_labes = title_labes + dataframe.columns[i] + '=' + str(round(dataframe[dataframe.columns[i]][0], 7)) 
            if (i != 7):
                title_labes = title_labes + ', '
    X = dataframe[label_changing_parameter_1]
    Y = dataframe[metric]
    Z = dataframe[label_changing_parameter_2]
    plt.scatter(X, Z, 80, Y, cmap='nipy_spectral')   #RdYlBu  #magma
    cbar=plt.colorbar()
    cbar.ax.set_ylabel(label_metric, rotation=90)
    #cbar.yaxis.tick_left()
    plt.xlabel(label_changing_parameter_1)
    plt.ylabel(label_changing_parameter_2)
    plt.title(title_labes, y=1.05, fontsize=10)
    plt.gcf().savefig(output_path + metric +'.png', dpi=300)
    plt.gcf().clear()

df = readout(directory)
if (mode=="2d"):
    plot_loss_2d(df, label_changing_parameter_1)
    plot_accuracy_2d(df, label_changing_parameter_1)
if (mode=="3d"):
    plot_3d(df, '$ Loss_{Training} $', 'loss',  label_changing_parameter_1, label_changing_parameter_2)
    plot_3d(df, '$ Loss_{Validation} $', 'val_loss', label_changing_parameter_1, label_changing_parameter_2)
    plot_3d(df, '$ Accuracy_{Training} $', 'accuracy', label_changing_parameter_1, label_changing_parameter_2)
    plot_3d(df, '$ Accuracy_{Validation} $', 'val_accuracy', label_changing_parameter_1, label_changing_parameter_2)
    
