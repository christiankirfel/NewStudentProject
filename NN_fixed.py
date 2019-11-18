#Loading the packages for running the networks
import os
import keras
import math
import sys
import matplotlib
import glob

###time measurement
from timeit import default_timer as timer
###

from keras.models import Sequential, Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, Activation
from keras import metrics
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sys import argv
#Loading the packages for handling the data
import uproot as ur
import pandas 
import numpy as np
#Loading packages needed for plottting
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator, MaxNLocator)
#Defining colours for the plots
#The colours were chosen using the xkcd guice
#color_tW = '#66FFFF'
color_tW = '#0066ff'
#color_tt = '#FF3333'
color_tt = '#990000'
color_sys = '#009900'
color_tW2 = '#02590f'
color_tt2 = '#FF6600'


my_path_to_data = '/cephfs/user/s6pinogg/PietBachelor/root_fixed_tZq/'
data_signal = my_path_to_data + 'tZq/'

data_background_diboson = my_path_to_data + 'diboson/'

data_background_ttZ = my_path_to_data + 'ttV/ttZ/'
data_background_ttW = my_path_to_data + 'ttV/ttW/'
data_background_ttH = my_path_to_data + 'ttV/ttH/'

data_background_ttbar = my_path_to_data + 'ttbar/'
data_background_tt2l = my_path_to_data + 'tt_2l/'

data_background_tWZ = my_path_to_data + 'singleTop/tWZ/'
data_background_4top = my_path_to_data + 'singleTop/4_top/'
data_background_tchannel = my_path_to_data + 'singleTop/tchannel/'
data_background_tW = my_path_to_data + 'singleTop/tW/'
data_background_nBoson = my_path_to_data + 'nBoson/'
data_background_ZJets = my_path_to_data + 'ZJets/'


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Setting up the output directories
output_path = '/cephfs/user/s6pinogg/PietBachelor/New_Samples/jobs/'
#output_path = './jobs/'
array_path = output_path + 'arrays/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(array_path):
    os.makedirs(array_path)

mysavedata = '/cephfs/user/s6pinogg/PietBachelor/Histo_Data/'



if not os.path.exists(mysavedata):
    os.makedirs(mysavedata)





#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#This is the main class for the adversarial neural network setup
class neuralNetworkEnvironment(object):

    def __init__(self):
        #At the moment not may variables are passed to the class. You might want to change this
        #A list of more general settings
       # self.variables = np.array(['m_b_jf','m_top','eta_jf','mT_W','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf'])
        self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','m_met','m_top','mT_W'])
        #self.variables = np.array(['eta_jf'])

        #The seed is used to make sure that both the events and the labels are shuffeled the same way because they are not inherently connected.
        self.seed = 250

        #All information necessary for the input
        #The exact data and targets are set later
        self.output_job = None 
        self.sample_training = None
        self.sample_validation = None
        self.target_training = None
        self.target_validation = None

        #Dimension of the variable input used to define the size of the first layer
        self.input_dimension = self.variables.shape
        #These arrays are used to save loss and accuracy of the two networks
        #That is also important to later be able to use the plotting software desired. matplotlib is not the best tool at all times
        self.discriminator_history_array = []
        self.model_history_array = []
        self.discriminator_history = None
        self.model = None
        self.network_input = None ###Change###
        #Here are the definitions for the two models
        #All information for the length of the training. Beware that epochs might only come into the pretraining
        #Iterations are used for the adversarial part of the training
        #If you want to make the training longer you want to change these numbers, there is no early stopping atm, feel free to add it
        self.discriminator_epochs = 350
        self.batchSize = 512
        #Setup of the networks, nodes and layers
        self.discriminator_layers = 3
        self.discriminator_nodes = 128
        #Setup of the networks, loss and optimisation
        self.my_optimizer = 'SGD'
        self.discriminator_lr = float(sys.argv[1])
        self.discriminator_momentum = 0.3
        self.discriminator_optimizer = SGD(lr = self.discriminator_lr, momentum = self.discriminator_momentum)
        self.discriminator_optimizer_adam = Adam(lr = self.discriminator_lr,beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay = self.discriminator_lr/self.discriminator_epochs,amsgrad=True)
        self.discriminator_dropout = 0.3
        self.discriminator_loss = 'binary_crossentropy'
        self.validation_fraction = 0.1

        self.output_job = output_path + 'epochs_%i/lr_%.2e/momentum_%.2f/' % (self.discriminator_epochs,self.discriminator_lr,self.discriminator_momentum)
        self.output_lr = output_path + 'epochs_%i/' % (self.discriminator_epochs)
        self.output_curve = self.output_lr + 'txtlr/'

        if not os.path.exists(self.output_job):
            os.makedirs(self.output_job)
        if not os.path.exists(self.output_curve):
            os.makedirs(self.output_curve)

        #The following set of variables is used to evaluate the result
        #fpr = false positive rate, tpr = true positive rate
        self.tpr = 0.  #true positive rate
        self.fpr = 0.  #false positive rate
        self.threshold = 0.
        self.auc = 0.  #Area under the curve

        ## Tree Sample Names MC
        self.signal_sample = "tHqLoop_nominal;1"
        self.background_sample = "tHqLoop_nominal;1"


    def read_root(self,pathtoMC,TreeName,BranchName):
        first_iteration = True
        SaveArray = np.array([])
        for files in ur.iterate(pathtoMC + '*.root',TreeName,BranchName,outputtype = pandas.DataFrame):
            files = files.to_numpy()
            if (first_iteration==True):

                SaveArray = files

            else:

                SaveArray = np.concatenate([SaveArray,files])
            
            first_iteration = False
        return SaveArray
    
    def ConArrays(self,arrays):
        conarray = np.concatenate(arrays)
        return conarray

    def initialize_sample(self):
        ### Signal 
        self.events_signal = self.read_root(data_signal,self.signal_sample,self.variables)
        self.weight_signal = self.read_root(data_signal,self.signal_sample,'weight_nominal') * 139
        ###Background
        ##Diboson
        self.events_background_diboson = self.read_root(data_background_diboson,self.background_sample,self.variables)
        self.weights_background_diboson = self.read_root(data_background_diboson,self.background_sample,'weight_nominal') * 139
        ##ttV
        self.events_background_ttZ = self.read_root(data_background_ttZ,self.background_sample,self.variables) 
        self.weights_background_ttZ = self.read_root(data_background_ttZ,self.background_sample,'weight_nominal') * 139

        self.events_background_ttW = self.read_root(data_background_ttW,self.background_sample,self.variables) 
        self.weights_background_ttW = self.read_root(data_background_ttW,self.background_sample,'weight_nominal') * 139

        self.events_background_ttH = self.read_root(data_background_ttH,self.background_sample,self.variables) 
        self.weights_background_ttH = self.read_root(data_background_ttH,self.background_sample,'weight_nominal') * 139

        self.events_background_ttV = self.ConArrays([self.events_background_ttZ,self.events_background_ttW,self.events_background_ttH])
        self.weights_background_ttV = self.ConArrays([self.weights_background_ttZ,self.weights_background_ttW,self.weights_background_ttH])
        ## SingleTop
        self.events_background_tWZ = self.read_root(data_background_tWZ,self.background_sample,self.variables)
        self.weights_background_tWZ = self.read_root(data_background_tWZ,self.background_sample,'weight_nominal') * 139

        self.events_background_4top = self.read_root(data_background_4top,self.background_sample,self.variables)
        self.weights_background_4top = self.read_root(data_background_4top,self.background_sample,'weight_nominal') * 139

        self.events_background_tchannel = self.read_root(data_background_tchannel,self.background_sample,self.variables)
        self.weights_background_tchannel = self.read_root(data_background_tchannel,self.background_sample,'weight_nominal') * 139

        self.events_background_tW = self.read_root(data_background_tW,self.background_sample,self.variables)
        self.weights_background_tW = self.read_root(data_background_tW,self.background_sample,'weight_nominal') * 139

        self.events_background_ST = self.ConArrays([self.events_background_tWZ,self.events_background_tW])
        self.weights_background_ST = self.ConArrays([self.weights_background_tWZ,self.weights_background_tW])
        ## ttbar 
        self.events_background_ttbar = self.read_root(data_background_ttbar,self.background_sample,self.variables)
        self.weights_background_ttbar = self.read_root(data_background_ttbar,self.background_sample,'weight_nominal') * 139

        self.events_background_tt2l = self.read_root(data_background_tt2l,self.background_sample,self.variables)
        self.weights_background_tt2l = self.read_root(data_background_tt2l,self.background_sample,'weight_nominal') * 139
        
        self.events_background_ttbar_all = self.ConArrays([self.events_background_ttbar,self.events_background_tt2l])
        self.weights_background_ttbar_all = self.ConArrays([self.weights_background_ttbar,self.weights_background_tt2l])

        ## Z plus Jets
        self.events_background_ZJets = self.read_root(data_background_ZJets,self.background_sample,self.variables)
        self.weights_background_ZJets = self.read_root(data_background_ZJets,self.background_sample,'weight_nominal') * 139
        ## All background put together
        self.events_background = self.ConArrays([self.events_background_diboson,self.events_background_ttV,self.events_background_ST,self.events_background_ttbar_all])
        self.weight_background = self.ConArrays([self.weights_background_diboson,self.weights_background_ttV,self.weights_background_ST,self.weights_background_ttbar_all])
        ## Check if the Event yields are correct 
        print('Events tZQ:' + '%.2f'%(self.weight_signal.sum()))
        print('Events diboson:' + '%.2f'%self.weights_background_diboson.sum())
        print('Events ttZ:' + '%.2f'%self.weights_background_ttZ.sum())
        print('Events ttW:' + '%.2f'%self.weights_background_ttW.sum())
        print('Events ttH:' + '%.2f'%self.weights_background_ttH.sum())
        print('Events tWZ:' + '%.2f'%self.weights_background_tWZ.sum())
        print('Events tW:' + '%.2f'%self.weights_background_tW.sum())
        print('Events ttbar:' + '%.2f'%self.weights_background_ttbar.sum())
        print('Events tt2_l:' + '%.2f'%self.weights_background_tt2l.sum())
        print('Events ZJets:' + '%.2f'%self.weights_background_ZJets.sum())


        #Reshaping the weights
        self.weight_signal = np.reshape(self.weight_signal, (len(self.events_signal), 1))
        self.weight_background = np.reshape(self.weight_background, (len(self.events_background), 1))

        #Calculating the weight ratio to scale the signal weight up. This tries to take the high amount of background into account
        self.weight_ratio = (self.weight_signal.sum())/ self.weight_background.sum()
        self.weight_signal = self.weight_signal / self.weight_ratio

        #Setting up the targets
        #target combined is used to make sure the systematics are seen as signal for the first net in the combined training
        self.target_signal = np.reshape([1 for x in range(len(self.events_signal))], (len(self.events_signal), 1))
        self.target_background = np.reshape([0 for x in range(len(self.events_background))], (len(self.events_background), 1))
        #The samples and corresponding targets are split into a sample for training and a sample for testing. Keep in mind that the same random seed should be used for both splits
        self.sample_training, self.sample_validation = train_test_split(np.concatenate((self.events_signal, self.events_background)), test_size = self.validation_fraction, random_state = self.seed)
        self.target_training, self.target_validation = train_test_split(np.concatenate((self.target_signal, self.target_background)), test_size = self.validation_fraction, random_state = self.seed)
        #Splitting the weights
        self.weight_training, self.weight_validation = train_test_split(np.concatenate((self.weight_signal, self.weight_background)), test_size = self.validation_fraction, random_state = self.seed)
        #Setting up a scaler
        #A scaler makes sure that all variables are normalised to 1 and have the same order of magnitude for that reason
        scaler = StandardScaler()
        self.sample_training = scaler.fit_transform(self.sample_training)
        self.sample_validation = scaler.fit_transform(self.sample_validation)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        ## Now we want to build our Neural Network
    def My_DiscrimatorBuild(self):
        self.model = Sequential()
        self.model.add(Dense(self.discriminator_nodes,input_shape=(self.input_dimension)))
        self.model.add(Activation('elu'))
        for layercount in range(self.discriminator_layers - 1):
            self.model.add(Dense(self.discriminator_nodes,activation = 'elu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.discriminator_dropout))
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.compile(loss=binary_crossentropy,metrics=['binary_accuracy'],weighted_metrics = [metrics.binary_accuracy],optimizer = self.discriminator_optimizer)
        self.model.summary()



    def trainDiscriminator(self):
        # With our Neural Network set up, we can now fit the training and test data
        self.discriminator_history = self.model.fit(self.sample_training, self.target_training.ravel(), epochs=self.discriminator_epochs, batch_size = self.batchSize, sample_weight = self.weight_training.ravel(), validation_data = (self.sample_validation, self.target_validation, self.weight_validation.ravel()))
        self.discriminator_history_array.append(self.discriminator_history)
        print(self.discriminator_history.history.keys())

    def predictModel(self):
        # Predict Model on 'unknown' data and analyze predicition with i.e ROC Curve and AUC Value
        
        self.model_prediction = self.model.predict(self.sample_training).ravel()
        self.model_prediction_test = self.model.predict(self.sample_validation).ravel()
        self.fpr, self.tpr, self.threshold = roc_curve(self.target_training, self.model_prediction)
        self.fpr_test, self.tpr_test, self.threshold_test = roc_curve(self.target_validation, self.model_prediction_test)
        self.auc = auc(self.fpr, self.tpr)
        self.auc_test = auc(self.fpr_test, self.tpr_test)

        print('Discriminator AUC Training:', self.auc)
        print('Discriminator AUC Test:', self.auc_test)

    def plotLosses(self):
        ax = plt.subplot(111)
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-1,2),useMathText=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(self.discriminator_history.history['loss'])
        plt.plot(self.discriminator_history.history['val_loss'])
        plt.title('Discriminator Losses')
        plt.ylabel('Loss',fontsize=12)
        plt.xlabel('Epoch',fontsize = 12)
        plt.legend(['train', 'test'], loc='upper left')
#        plt.legend(loc="upper right", prop={'size' : 7})
        plt.gcf().savefig(self.output_job + 'losses.png')
        plt.gcf().clear()

    def plotRoc(self):
        plt.title('Receiver Operating Characteristic')
        plt.plot(self.fpr, self.tpr, 'g--',color='blue', label='$AUC_{train}$ = %0.2f'% self.auc)
        plt.plot(self.fpr_test, self.tpr_test, 'g--',color='orange', label='$AUC_{test}$ = %0.2f'% self.auc_test)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.,1.])
        plt.ylim([-0.,1.])
        plt.ylabel('True Positive Rate', fontsize='large')
        plt.xlabel('False Positive Rate', fontsize='large')
        plt.legend(frameon=False)
        #plt.show()
        plt.gcf().savefig(self.output_job + 'roc.png')
        plt.gcf().clear()   

    def plotSeparation(self):
        self.signal_histo = []
        self.background_histo = []
        for i in range(len(self.sample_validation)):
            if self.target_validation[i] == 1:
                self.signal_histo.append(self.model_prediction_test[i])
            if self.target_validation[i] == 0:
                self.background_histo.append(self.model_prediction_test[i])
                
        plt.hist(self.signal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step",density = True,color=color_tW, label = "Signal")
        plt.hist(self.background_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tt, label = "Background")
        plt.legend()
        plt.xlabel('Network response', horizontalalignment='left', fontsize='large')
        plt.ylabel('Event fraction', fontsize='large')
        plt.legend(frameon=False)
        plt.gcf().savefig(self.output_job + 'separation.png')
        plt.gcf().clear()

    def plotWeightedAccuracy(self):
        ax = plt.subplot(111)
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-1,2),useMathText=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(self.discriminator_history.history['weighted_binary_accuracy'])
        plt.plot(self.discriminator_history.history['val_weighted_binary_accuracy'])
        plt.title('model weighted accuracy')
        plt.ylabel('weighted accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.gcf().savefig(self.output_job + 'weighted_acc.png')
        plt.gcf().clear()

    def plotAccuracy(self):
        ax = plt.subplot(111)
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-2,2),useMathText=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.plot(self.discriminator_history.history['binary_accuracy'])
        plt.plot(self.discriminator_history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.gcf().savefig(self.output_job + 'acc.png')
        plt.gcf().clear()

    def ParamstoTxt(self):
        paramsfile = (self.output_job + 'params.txt')
        file = open(paramsfile,'w')
        file.write('We are using the %s optimizer at the moment' % (self.my_optimizer))
        file.write('\n')
        file.write('Number of epochs: %i' % (self.discriminator_epochs))
        file.write('\n')
        file.write('Batch Size: %i' % (self.batchSize))
        file.write('\n')
        file.write('Number of hidden layers: %i' % (self.discriminator_layers))
        file.write('\n')
        file.write('Number of nodes: %i' % (self.discriminator_nodes))
        file.write('\n') 
        file.write('Dropout: %.2f' % (self.discriminator_dropout))
        file.write('\n')
        file.write('Validation fraction: %.2f' % (self.validation_fraction))
        file.write('\n')
        file.write('Learning rate:%.3e' % (self.discriminator_lr))
        file.write('\n')
        file.write('Momentum:%.2f' % (self.discriminator_momentum))
        file.write('\n')
        file.write('Loss after %i epochs: %.5e' % (self.discriminator_epochs,self.discriminator_history.history['loss'][-1]))
        file.write('\n')
        file.write('Validation Loss after %i epochs: %.5e' % (self.discriminator_epochs,self.discriminator_history.history['val_loss'][-1]))
        file.write('\n')
        file.write('Discriminator AUC:%.3f'%self.auc)
        file.write('\n')
        file.close()

            

        file = open(self.output_curve + 'lrcurve_%.1e.txt'%(self.discriminator_lr),'w')
        file.write('%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e'%(self.discriminator_history.history['loss'][-1],self.discriminator_history.history['val_loss'][-1],self.discriminator_lr,self.auc,self.discriminator_history.history['binary_accuracy'][-1],self.discriminator_history.history['val_binary_accuracy'][-1],self.discriminator_momentum))
        file.close()

    def plot_lr(self,filelist):
        with open(self.output_lr + 'plot_lr.txt','w') as self.outfile:
            for fname in filelist:
                with open(self.output_curve + fname,'r') as self.infile:
                    self.outfile.write(self.infile.read())
                    self.outfile.write('\n')
        lr_list = []
        val_loss_plot = []
        loss_plot = []
        auc_list = []
        acc_list= []
        val_acc_list= []
        momentum_list= []
        self.outfile = self.output_lr + 'plot_lr.txt'
        results = open(self.outfile,'r')
        for line in results:
            dataline = line
            data = dataline.split(',')
            lr_list.append(float(data[2]))
            val_loss_plot.append(float(data[1]))
            loss_plot.append(float(data[0]))
            auc_list.append(float(data[3]))
            acc_list.append(float(data[4]))
            val_acc_list.append(float(data[5]))
            momentum_list.append(float(data[6]))
        lr_list = np.array(lr_list)
        val_loss_plot = np.array(val_loss_plot)
        loss_plot = np.array(loss_plot)
        auc_list = np.array(auc_list)
        val_acc_list = np.array(val_acc_list)
        acc_list = np.array(acc_list)
        momentum_list = np.array(momentum_list)

        ### Plot Lists
        ax = plt.subplot(111)
        plt.plot(lr_list,loss_plot,color = color_tt,label='Training',marker = 'x',linestyle = 'None')
        plt.plot(lr_list,val_loss_plot,color = color_tW,label = 'Test',marker = 'x',linestyle = 'None')
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-2,2),useMathText = True)
        ax.set_xscale("log", nonposx='clip')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.legend()
        plt.gcf().savefig(self.output_lr+'LRPlot'+'_'+ '%.2f.png'%self.discriminator_momentum)
        plt.gcf().clear()

        ax = plt.subplot(111)
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-2,2),useMathText = True)
        ax.set_xscale("log", nonposx='clip')
        plt.plot(lr_list,auc_list, color = 'navy', marker = 'x', linestyle = 'None',label='AUC Value')
        plt.plot(lr_list,acc_list,color="red",marker='x',linestyle='None',label='Accuracy')
        plt.legend()
        plt.xlabel('Learning rate')
        plt.gcf().savefig(self.output_lr+'LRAucPlot'+'_'+ '%.2f.png' % (self.discriminator_momentum))
        plt.gcf().clear()



        ###
            
        



    def HistObject(self,Xaxisbins,Yaxisbins,range1,range2,bins,labelxaxis,savelabel,numbervariable):
        self.events_background_ttZ_tWZ = np.concatenate([self.events_background_ttZ,self.events_background_tWZ])
        self.events_background_ttbar_tW = np.concatenate([self.events_background_ttbar_all,self.events_background_tW])

        self.hist_tZq = self.events_signal.transpose() 
        self.hist_diboson = self.events_background_diboson.transpose() 
        self.hist_ttbar_tW = self.events_background_ttbar_tW.transpose()
        self.hist_ttZ_tWZ = self.events_background_ttZ_tWZ.transpose()
        self.hist_ZJets = self.events_background_ZJets.transpose()

        self.hist_tZq = np.clip(self.hist_tZq,None,range2)
        self.hist_diboson = np.clip(self.hist_diboson,None,range2)
        self.hist_ttbar_tW = np.clip(self.hist_ttbar_tW,None,range2)
        self.hist_ttZ_tWZ = np.clip(self.hist_ttZ_tWZ,None,range2)
        self.hist_ZJets = np.clip(self.hist_ZJets,None,range2)

        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.locator_params(axis='x', nbins=Xaxisbins)
        plt.locator_params(axis='y', nbins=Yaxisbins)

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.tick_params(direction='in',which='minor', length=2)

        plt.hist(self.hist_tZq[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.hist_diboson[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='royalblue',label='diboson',density = True)
        plt.hist(self.hist_ttbar_tW[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='red',label=r'$t\bar{t}+tW$',density = True)
        plt.hist(self.hist_ttZ_tWZ[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='lime',label=r'$ttZ+tWZ$',density = True)
        plt.hist(self.hist_ZJets[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='orange',label=r'$Z+Jets$',density = True)

        plt.legend(frameon = False)

        plt.xlim(range1,range2)

        plt.xlabel(labelxaxis,horizontalalignment='right',x=1.0,fontsize=12)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)

        plt.gcf().savefig(mysavedata + savelabel +'.png')
        plt.gcf().clear()

    def Runtime(self,start,stop):
        file = open(self.output_job + 'params.txt','a')
        file.write('Runtime of program: %.2f seconds' % (stop-start))
        file.close()

start = timer()

training = neuralNetworkEnvironment()
training.initialize_sample()
training.My_DiscrimatorBuild()
training.trainDiscriminator()
training.predictModel()
training.plotLosses()
training.plotRoc()
training.plotSeparation()
training.plotWeightedAccuracy()
training.plotAccuracy()
training.ParamstoTxt()

###
#self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','m_met','m_top','mT_W'])
#training.HistObject(10,5,0,800,16,'$m(bj_F)$','m_b_jf',0)
#training.HistObject(10,5,0,5,15,'$\eta(j_f)$','eta_jf',1)
#training.HistObject(5,10,-2.5,2.5,5,'$q(l^W)$','q_lW',2)
#training.HistObject(6,3,0,3.,12,'$\eta(l^W)$','eta_lW',3)
#training.HistObject(6,6,0,300.,11,'$p_T(W)$[GeV]','pT_W',4)
#training.HistObject(10,4,0,200.,11,'$p_T(l^W)$[GeV]','pT_lW',5)
#training.HistObject(6,5,60.,120.,25,'$m(ll)$[GeV]','m_Z',6)
#training.HistObject(10,4,0,5.,11,'$\eta(Z)$','eta_Z',7)
#training.HistObject(7,5,0,7.,16,'$\Delta R(j_f,Z)$','dR_jf_Z',8)
#training.HistObject(6,5,0,300.,11,'$p_T(j_f)$[GeV]','pT_jf',9)
#training.HistObject(10,6,0,200.,11,'$p_T(j_r)$[GeV]','pT_jr',10)
#training.HistObject(10,6,0,5.,16,'$\eta(j_r)$','eta_jr',11)
#training.HistObject(6,6,0,300.,11,'$p_T(Z)$[GeV]','pT_Z',12)
#training.HistObject(6,6,0,600,31,'$E^{miss}$[GeV]','m_met',13)
#training.HistObject(6,6,0,600,31,'$m_t$[GeV]','m_top',14)
#training.HistObject(12,6,0,240,8,'$m_t$[GeV]','mT_W',15)
###

text_files = [f for f in os.listdir(training.output_curve) if f.endswith('.txt')]
training.plot_lr(text_files)

end = timer()
training.Runtime(start,end)