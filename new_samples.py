#Loading the packages for running the networks
import os
import keras
import math
import sys
import matplotlib

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
#Defining colours for the plots
#The colours were chosen using the xkcd guice
#color_tW = '#66FFFF'
color_tW = '#0066ff'
#color_tt = '#FF3333'
color_tt = '#990000'
color_sys = '#009900'
color_tW2 = '#02590f'
color_tt2 = '#FF6600'

ax = plt.subplot(111)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Setting up the output directories
output_path = '/cephfs/user/s6pinogg/PietBachelor/New_Samples/jobs/'
#output_path = './jobs/'
array_path = output_path + 'arrays/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(array_path):
    os.makedirs(array_path)







#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#This is the main class for the adversarial neural network setup
class neuralNetworkEnvironment(object):

    def __init__(self):
        #At the moment not may variables are passed to the class. You might want to change this
        #A list of more general settings
       # self.variables = np.array(['m_b_jf','m_top','eta_jf','mT_W','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf'])
        self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','pT_lW'])
        #The seed is used to make sure that both the events and the labels are shuffeled the same way because they are not inherently connected.
        self.seed = 250
        #All information necessary for the input
        #The exact data and targets are set later
        self.output_job = None 
        self.input_path_sample = "/cephfs/user/s6pinogg/PietBachelor/signal_tZq/tZq/mc16a.412063.aMCPy8EG_tllq_nf4.FS.nominal.root"
        self.input_path_background = "/cephfs/user/s6pinogg/PietBachelor/background_NN/nBosons/mc16a.364253.Sh222_lllv.FS.nominal.root"
        self.input_path_background_2 = "/cephfs/user/s6pinogg/PietBachelor/background_NN/nBosons/mc16a.364250.Sh222_llll.FS.nominal.root" ## NEW
        self.input_path_background_3 = "/cephfs/user/s6pinogg/PietBachelor/18-10_tZVar/ttV/ttbar/mc16a.410470.PhPy8EG_ttbar_hdamp258p75_l.FS.nominal.root"## NEW

        self.signal_sample = "tHqLoop_nominal"
        self.background_sample = "tHqLoop_nominal"
        self.signal_tree = ur.open(self.input_path_sample)[self.signal_sample]
        self.background_tree = ur.open(self.input_path_background)[self.background_sample]
        self.background_tree_2 = ur.open(self.input_path_background_2)[self.background_sample] ## NEW 
        self.background_tree_3 = ur.open(self.input_path_background_3)[self.background_sample] ## NEW 

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
        self.discriminator_epochs = 200
        self.batchSize = 64
        #Setup of the networks, nodes and layers
        self.discriminator_layers = 3
        self.discriminator_nodes = 128
        #Setup of the networks, loss and optimisation
        ## just an integer
        self.queue = 3
        ##
        self.my_optimizer = 'Adam'
        self.discriminator_lr = float(sys.argv[1])
        self.discriminator_momentum = 0.9
        self.discriminator_optimizer = SGD(lr = self.discriminator_lr, momentum = self.discriminator_momentum)
        self.discriminator_optimizer_adam = Adam(lr = self.discriminator_lr)
        self.discriminator_dropout = 0.3
        self.discriminator_loss = binary_crossentropy
        self.validation_fraction = 0.05

        self.output_job = output_path + 'epochs_%i/lr_%.3f/momentum_%.3f/' % (self.discriminator_epochs,self.discriminator_lr,self.discriminator_momentum)
        self.output_lr = output_path + 'epochs_%i/' % (self.discriminator_epochs)
        
        if not os.path.exists(self.output_job):
            os.makedirs(self.output_job)

        ###
        #The following set of variables is used to evaluate the result
        #fpr = false positive rate, tpr = true positive rate
        self.tpr = 0.  #true positive rate
        self.fpr = 0.  #false positive rate
        self.threshold = 0.
        self.auc = 0.  #Area under the curve

        #


#Initializing the data and target samples
#The split function cuts into a training sample and a test sample
#Important note: Have to use the same random seed so that event and target stay in the same order as we shuffle
    def initialize_sample(self):
        #Signal and background are needed for the classification task, signal and systematic for the adversarial part
        #In this first step the events are retrieved from the tree, using the chosen set of variables
        #The numpy conversion is redundant
        #print(self.background_tree.pandas.df("m_top").to_numpy())
        #print(self.signal_tree.pandas.df("m_top").to_numpy())

        self.events_signal = self.signal_tree.pandas.df(self.variables).to_numpy()
        self.events_background = np.concatenate([self.background_tree.pandas.df(self.variables).to_numpy(),self.background_tree_2.pandas.df(self.variables).to_numpy(),self.background_tree_3.pandas.df(self.variables).to_numpy()])
        #Setting up the weights. The weights for each tree are stored in 'weight_nominal'
        self.weight_signal = self.signal_tree.pandas.df('weight_nominal').to_numpy()
        self.weight_background = np.concatenate([self.background_tree.pandas.df('weight_nominal').to_numpy(),self.background_tree_2.pandas.df('weight_nominal').to_numpy(),self.background_tree_3.pandas.df('weight_nominal').to_numpy()])
        #Reshaping the weights
        self.weight_signal = np.reshape(self.weight_signal, (len(self.events_signal), 1))
        self.weight_background = np.reshape(self.weight_background, (len(self.events_background), 1))
        #Normalisation to the eventcount can be used instead of weights, especially if using data
        self.norm_signal = np.reshape([1./float(len(self.events_signal)) for x in range(len(self.events_signal))], (len(self.events_signal), 1))
        self.norm_background = np.reshape([1./float(len(self.events_background)) for x in range(len(self.events_background))], (len(self.events_background), 1))
        #Calculating the weight ratio to scale the signal weight up. This tries to take the high amount of background into account
        self.weight_ratio = ( self.weight_signal.sum())/ self.weight_background.sum()
        self.weight_signal = self.weight_signal / self.weight_ratio

        #Setting up the targets
        #target combined is used to make sure the systematics are seen as signal for the first net in the combined training
        self.target_signal = np.reshape([1 for x in range(len(self.events_signal))], (len(self.events_signal), 1))
        self.target_background = np.reshape([0 for x in range(len(self.events_background))], (len(self.events_background), 1))
        #The samples and corresponding targets are now split into a sample for training and a sample for testing. Keep in mind that the same random seed should be used for both splits
        self.sample_training, self.sample_validation = train_test_split(np.concatenate((self.events_signal, self.events_background)), test_size = self.validation_fraction, random_state = self.seed)
        self.target_training, self.target_validation = train_test_split(np.concatenate((self.target_signal, self.target_background)), test_size = self.validation_fraction, random_state = self.seed)
        #Splitting the weights
        self.weight_training, self.weight_validation = train_test_split(np.concatenate((self.weight_signal, self.weight_background)), test_size = self.validation_fraction, random_state = self.seed)
        self.norm_training, self.norm_validation = train_test_split(np.concatenate((self.norm_signal, self.norm_background)), test_size = self.validation_fraction, random_state = self.seed)

        #Setting up a scaler
        #A scaler makes sure that all variables are normalised to 1 and have the same order of magnitude for that reason
        scaler = StandardScaler()
        self.sample_training = scaler.fit_transform(self.sample_training)
        self.sample_validation = scaler.fit_transform(self.sample_validation)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

    def My_DiscrimatorBuild(self):
        #Understanding Keras
        self.model = Sequential()
        self.model.add(Dense(self.discriminator_nodes,input_shape=(self.input_dimension)))
        self.model.add(Activation('elu'))
        for layercount in range(self.discriminator_layers - 1):
            self.model.add(Dense(self.discriminator_nodes,activation = 'relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.discriminator_dropout))
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',weighted_metrics = [metrics.binary_accuracy],optimizer = self.discriminator_optimizer_adam)
        #self.model.compile(loss='binary_crossentropy',weighted_metrics =[metrics.binary_accuracy],optimizer = self.discriminator_optimizer)
        #self.model.summary()
        

    def trainDiscriminator(self):

        #print(self.target_training[12:500])
        #print(self.target_training[-1:-100])

        #self.model_discriminator.summary()

        self.discriminator_history = self.model.fit(self.sample_training, self.target_training.ravel(), epochs=self.discriminator_epochs, batch_size = self.batchSize, sample_weight = self.weight_training.ravel(), validation_data = (self.sample_validation, self.target_validation, self.weight_validation.ravel()))
        #self.discriminator_history = self.model_discriminator.fit(self.sample_training, self.target_training.ravel(), epochs=self.discriminator_epochs, batch_size = self.batchSize, sample_weight = self.weight_training.ravel(), validation_data = (self.sample_validation, self.target_validation, self.weight_validation.ravel()))
        self.discriminator_history_array.append(self.discriminator_history)
        #print(self.discriminator_histoax = plt.subplot(111)ry.history.keys())


    # Compile and fit the Neural Network with different learning rate to optimize result



    def predictModel(self):

        self.model_prediction = self.model.predict(self.sample_training).ravel()
        self.model_prediction_test = self.model.predict(self.sample_validation).ravel()
        self.fpr, self.tpr, self.threshold = roc_curve(self.target_training, self.model_prediction)
        self.fpr_test, self.tpr_test, self.threshold_test = roc_curve(self.target_validation, self.model_prediction_test)
        self.auc = auc(self.fpr, self.tpr)
        self.auc_test = auc(self.fpr_test, self.tpr_test)


        print('Discriminator AUC:', self.auc)

    def plotLosses(self,learning_rate):
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-1,2))
        
        plt.plot(self.discriminator_history.history['loss'])
        plt.plot(self.discriminator_history.history['val_loss'])
        plt.title('Discriminator Losses with L_r=%.5f,m=%.3f,%i Epoch' % (learning_rate,self.discriminator_momentum,self.discriminator_epochs))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
#        plt.legend(loc="upper right", prop={'size' : 7})
        plt.gcf().savefig(self.output_job + 'losses.png')
        plt.gcf().clear()




    def plotRoc(self,learning_rate):
        plt.title('Receiver Operating Characteristic with L_r=%.5f,m=%.3f,%i Epoch' % (learning_rate,self.discriminator_momentum,self.discriminator_epochs))
        plt.plot(self.fpr, self.tpr, 'g--', label='$AUC_{train}$ = %0.2f'% self.auc)
        plt.plot(self.fpr_test, self.tpr_test, 'g--',color ='r', label='$AUC_{test}$ = %0.2f'% self.auc_test)
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

    def plotSeparation(self,learning_rate):
        self.signal_histo = []
        self.background_histo = []
        for i in range(len(self.sample_validation)):
            if self.target_validation[i] == 1:
                self.signal_histo.append(self.model_prediction[i])
            if self.target_validation[i] == 0:
                self.background_histo.append(self.model_prediction[i])
                
        plt.hist(self.signal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tW, label = "Signal")
        plt.hist(self.background_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tt, label = "Background")
        plt.title('with L_r=%.5f,m=%.3f,%i Epoch' % (learning_rate,self.discriminator_momentum,self.discriminator_epochs))
        plt.legend()
        plt.xlabel('Network response', horizontalalignment='left', fontsize='large')
        plt.ylabel('Event fraction', fontsize='large')
        plt.legend(frameon=False)
        plt.gcf().savefig(self.output_job + 'separation.png')
        plt.gcf().clear()



 
    def plotAccuracy(self,learning_rate):
        plt.plot(self.discriminator_history.history['weighted_binary_accuracy'])
        plt.plot(self.discriminator_history.history['val_weighted_binary_accuracy'])
        plt.title('model accuracy with L_r=%.3f,m=%.3f,%i Epoch' % (learning_rate,self.discriminator_momentum,self.discriminator_epochs))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
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
        #print(self.discriminator_history.history('loss')[-1])
        file.write('Loss after %i epochs: %.5e' % (self.discriminator_epochs,self.discriminator_history.history['loss'][-1]))
        file.write('\n')
        file.write('Validation Loss after %i epochs: %.5e' % (self.discriminator_epochs,self.discriminator_history.history['val_loss'][-1]))
        file.write('\n')
        file.write('Discriminator AUC:%.3f'%self.auc)
        file.write('\n')
        file.close()

        file = open(self.output_job + 'loss.txt','w')
        for i in range(len(self.discriminator_history.history['loss'])-1):
            file.write('%.5e,%.5e' % (self.discriminator_history.history['loss'][i],self.discriminator_history.history['val_loss'][i]))
            file.write('\n')
            
        file.close()

        file = open(self.output_lr + 'lrcurve_0_%i.txt'%(self.discriminator_lr*10000),'w')
        file.write('%.4e,%.4e,%.4e,%.4e'%(self.discriminator_history.history['loss'][-1],self.discriminator_history.history['val_loss'][-1],self.discriminator_lr,self.auc))
        file.close()
    def plot_lr(self,filelist):
        with open(self.output_lr + 'plot_lr.txt','w') as self.outfile:
            for fname in filelist:
                with open(fname,'r') as self.infile:
                    self.outfile.write(self.infile.read())
                    self.outfile.write('\n')
        lr_list = []
        val_loss_plot = []
        loss_plot = []
        auc_list = []
        self.outfile = self.output_lr + 'plot_lr.txt'
        results = open(self.outfile,'r')
        for line in results:
            dataline = line
            data = dataline.split(',')
            lr_list.append(float(data[2]))
            val_loss_plot.append(float(data[1]))
            loss_plot.append(float(data[0]))
            auc_list.append(float(data[3]))

        ### Plot Lists
        plt.plot(lr_list,loss_plot,color = color_tt,label='Training',marker = 'x')
        plt.plot(lr_list,val_loss_plot,color = color_tW,label = 'Test',marker = 'x')
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-1,2),useMathText = True)
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Loss Curve')
        plt.legend()
        plt.gcf().savefig(self.output_lr+'LRPlot.png')
        plt.gcf().clear()
        ###
            
            
        

    def Runtime(self,start,stop):
        file = open(self.output_job + 'params.txt','a')
        file.write('Runtime of program: %.2f seconds' % (stop-start))
        file.close()



#In the following options and variables are read in
#This is done to keep the most important features clearly represented

#You need to find a nice set of variables to use and add them as a text file
#with open('/cephfs/user/s6chkirf/whk_ANN_variables.txt','r') as varfile:binary_crossentropy"
  #  variableList = varfile.read().splitlines() binary_crossentropy"
#binary_crossentropy"
#print(variableList)binary_crossentropy"
#def ReadOptions(region):
#    with variables = variableListopen('/cephfs/user/s6chkirf/config_whk_ANN.txt','r') as infile:
#        optionsList = infile.read().splitlines()
#    OptionDict = dict()
#    for options in optionsList:
#		# definition of a comment
#        if options.startswith('#'): continue        plt.ylabel('Training Loss')
#        templist = options.split(' ')
#        if len(templist) == 2:
#            OptionDict[templist[0]] = templist[1]
#        else:
#            OptionDict[templist[0]] = templist[1:]
#    return OptionDict
#    # a dictionary of options is returned

start = timer()

first_training = neuralNetworkEnvironment()

first_training.initialize_sample()
first_training.My_DiscrimatorBuild()
first_training.trainDiscriminator()
first_training.predictModel()
first_training.plotRoc(first_training.discriminator_lr)
first_training.plotSeparation(first_training.discriminator_lr)
first_training.plotAccuracy(first_training.discriminator_lr)
first_training.plotLosses(first_training.discriminator_lr)
first_training.ParamstoTxt()
"""
### For LR Plot
filenames = []
for number in range(10):
    filenames.append(first_training.output_lr+'lrcurve_%i.txt' % (number+1))
for number in range(28):
    filenames.append(first_training.output_lr+'lrcurve_0_%i.txt' % (number+1))
first_training.plot_lr(filenames)
"""
###
end = timer()
first_training.Runtime(start,end)

