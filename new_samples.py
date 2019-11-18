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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
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
my_path_to_data = '/cephfs/user/s6pinogg/PietBachelor/tZq_plus_backgrounds/'
data_background_diboson = my_path_to_data + 'diboson/'
data_background_ttV = my_path_to_data + 'ttV/'
data_background_ttbar = my_path_to_data + 'ttbar/'
data_background_tWZ = my_path_to_data + 'singleTop/'

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
        self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','m_met','m_top'])
        #self.variables = np.array(['eta_jf'])

        #The seed is used to make sure that both the events and the labels are shuffeled the same way because they are not inherently connected.
        self.seed = 250
        #All information necessary for the input
        #The exact data and targets are set later
        self.output_job = None 

        ### tW ttbar
        """
        self.variables = np.array(["mass_lep1jet2", "pTsys_lep1lep2met", "pTsys_jet1jet2", "mass_lep1jet1", "deltapT_lep1_jet1", "deltaR_lep1_jet2", "deltaR_lep1lep2_jet2", "mass_lep2jet1", "pT_jet2", "deltaR_lep1_jet1", "deltaR_lep1lep2_jet1jet2met", "deltaR_lep2_jet2", "cent_lep2jet2", "deltaR_lep2_jet1"])
        self.input_path_sample = "/cephfs/user/s6chkirf/work/area/run/test_ANNinput.root"
        self.input_path_background = self.input_path_sample
        self.signal_sample = "wt_nominal"
        self.background_sample = "tt_nominal"
        """
        ###


        ###Signal tZq input files 2015-2018
        self.input_path_sample_2018 = '/cephfs/user/s6pinogg/PietBachelor/tZq_plus_backgrounds/tZq/mc16e.412063.aMCPy8EG_tllq_nf4.FS.nominal.root'
        self.input_path_sample_2017 = '/cephfs/user/s6pinogg/PietBachelor/tZq_plus_backgrounds/tZq/mc16d.412063.aMCPy8EG_tllq_nf4.FS.nominal.root'
        self.input_path_sample_15_16 = '/cephfs/user/s6pinogg/PietBachelor/tZq_plus_backgrounds/tZq/mc16a.412063.aMCPy8EG_tllq_nf4.FS.nominal.root'
        ### Background diboson input files
        ## 2018
        self.input_path_background_diboson_2018_ZqqZvv = data_background_diboson + 'mc16e.363355.Sh221_ZqqZvv.FS.nominal.root'
        self.input_path_background_diboson_2018_ZqqZll = data_background_diboson + 'mc16e.363356.Sh221_ZqqZll.FS.nominal.root'
        self.input_path_background_diboson_2018_WqqZvv = data_background_diboson + 'mc16e.363357.Sh221_WqqZvv.FS.nominal.root'
        self.input_path_background_diboson_2018_WqqZll = data_background_diboson + 'mc16e.363358.Sh221_WqqZll.FS.nominal.root'
        self.input_path_background_diboson_2018_WpqqWmlv = data_background_diboson + 'mc16e.363359.Sh221_WpqqWmlv.FS.nominal.root'
        self.input_path_background_diboson_2018_WplvWmqq = data_background_diboson + 'mc16e.363360.Sh221_WplvWmqq.FS.nominal.root'
        self.input_path_background_diboson_2018_WlvZqq = data_background_diboson + 'mc16e.363489.Sh221_WlvZqq.FS.nominal.root'
        self.input_path_background_diboson_2018_llll = data_background_diboson + 'mc16e.364250.Sh222_llll.FS.nominal.root'
        self.input_path_background_diboson_2018_lllv = data_background_diboson + 'mc16e.364253.Sh222_lllv.FS.nominal.root'
        self.input_path_background_diboson_2018_llvv = data_background_diboson + 'mc16e.364254.Sh222_llvv.FS.nominal.root'        
        self.input_path_background_diboson_2018_lvvv = data_background_diboson + 'mc16e.364255.Sh222_lvvv.FS.nominal.root'        
        self.input_path_background_diboson_2018_llll_lowMll = data_background_diboson + 'mc16e.364288.Sh222_llll_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2018_lllv_lowMll = data_background_diboson + 'mc16e.364289.Sh222_lllv_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2018_llvv_lowMll = data_background_diboson + 'mc16e.364290.Sh222_llvv_lowMllPtComp.FS.nominal.root'        
        ## 2017
        self.input_path_background_diboson_2017_ZqqZll = data_background_diboson + 'mc16d.363356.Sh221_ZqqZll.FS.nominal.root'
        self.input_path_background_diboson_2017_WqqZvv = data_background_diboson + 'mc16d.363357.Sh221_WqqZvv.FS.nominal.root'
        self.input_path_background_diboson_2017_WqqZll = data_background_diboson + 'mc16d.363358.Sh221_WqqZll.FS.nominal.root'
        self.input_path_background_diboson_2017_WpqqWmlv = data_background_diboson + 'mc16d.363359.Sh221_WpqqWmlv.FS.nominal.root'
        self.input_path_background_diboson_2017_WplvWmqq = data_background_diboson + 'mc16d.363360.Sh221_WplvWmqq.FS.nominal.root'
        self.input_path_background_diboson_2017_WlvZqq = data_background_diboson + 'mc16d.363489.Sh221_WlvZqq.FS.nominal.root'
        self.input_path_background_diboson_2017_llll = data_background_diboson + 'mc16d.364250.Sh222_llll.FS.nominal.root'
        self.input_path_background_diboson_2017_lllv = data_background_diboson + 'mc16d.364253.Sh222_lllv.FS.nominal.root'
        self.input_path_background_diboson_2017_llvv = data_background_diboson + 'mc16d.364254.Sh222_llvv.FS.nominal.root'        
        self.input_path_background_diboson_2017_lvvv = data_background_diboson + 'mc16d.364255.Sh222_lvvv.FS.nominal.root'        
        self.input_path_background_diboson_2017_llll_lowMll = data_background_diboson + 'mc16d.364288.Sh222_llll_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2017_lllv_lowMll = data_background_diboson + 'mc16d.364289.Sh222_lllv_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2017_llvv_lowMll = data_background_diboson + 'mc16d.364290.Sh222_llvv_lowMllPtComp.FS.nominal.root' 

        ## 2015-2016
        self.input_path_background_diboson_2016_ZqqZvv = data_background_diboson + 'mc16a.363355.Sh221_ZqqZvv.FS.nominal.root'
        self.input_path_background_diboson_2016_ZqqZll = data_background_diboson + 'mc16a.363356.Sh221_ZqqZll.FS.nominal.root'
        self.input_path_background_diboson_2016_WqqZvv = data_background_diboson + 'mc16a.363357.Sh221_WqqZvv.FS.nominal.root'
        self.input_path_background_diboson_2016_WqqZll = data_background_diboson + 'mc16a.363358.Sh221_WqqZll.FS.nominal.root'
        self.input_path_background_diboson_2016_WpqqWmlv = data_background_diboson + 'mc16a.363359.Sh221_WpqqWmlv.FS.nominal.root'
        self.input_path_background_diboson_2016_WplvWmqq = data_background_diboson + 'mc16a.363360.Sh221_WplvWmqq.FS.nominal.root'
        self.input_path_background_diboson_2016_WlvZqq = data_background_diboson + 'mc16a.363489.Sh221_WlvZqq.FS.nominal.root'
        self.input_path_background_diboson_2016_llll = data_background_diboson + 'mc16a.364250.Sh222_llll.FS.nominal.root'
        self.input_path_background_diboson_2016_lllv = data_background_diboson + 'mc16a.364253.Sh222_lllv.FS.nominal.root'
        self.input_path_background_diboson_2016_llvv = data_background_diboson + 'mc16a.364254.Sh222_llvv.FS.nominal.root'        
        self.input_path_background_diboson_2016_lvvv = data_background_diboson + 'mc16a.364255.Sh222_lvvv.FS.nominal.root'        
        self.input_path_background_diboson_2016_llll_lowMll = data_background_diboson + 'mc16a.364288.Sh222_llll_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2016_lllv_lowMll = data_background_diboson + 'mc16a.364289.Sh222_lllv_lowMllPtComp.FS.nominal.root'        
        self.input_path_background_diboson_2016_llvv_lowMll = data_background_diboson + 'mc16a.364290.Sh222_llvv_lowMllPtComp.FS.nominal.root' 

        ### Background ttbarZ -> ee
        ## 2018
        self.input_path_background_ttZ_2018_ee = data_background_ttV + 'mc16e.410218.aMCPy8EG_ttee.FS.nominal.root'
        ## 2017 
        self.input_path_background_ttZ_2017_ee = data_background_ttV + 'mc16d.410218.aMCPy8EG_ttee.FS.nominal.root'
        ## 2015-2016
        self.input_path_background_ttZ_2016_ee = data_background_ttV + 'mc16a.410218.aMCPy8EG_ttee.FS.nominal.root'
        ### Background ttbar -> ll
        ## 2018
        self.input_path_background_ttZ_2018_nunu = data_background_ttV + 'mc16e.410156.aMCPy8EG_ttZnunu.FS.nominal.root'
        self.input_path_background_ttZ_2018_mumu = data_background_ttV + 'mc16e.410219.aMCPy8EG_ttmumu.FS.nominal.root'
        self.input_path_background_ttZ_2018_tautau = data_background_ttV + 'mc16e.410220.aMCPy8EG_tttautau.FS.nominal.root'
        ## 2017
        self.input_path_background_ttZ_2017_nunu = data_background_ttV + 'mc16d.410156.aMCPy8EG_ttZnunu.FS.nominal.root'
        self.input_path_background_ttZ_2017_mumu = data_background_ttV + 'mc16d.410219.aMCPy8EG_ttmumu.FS.nominal.root'
        self.input_path_background_ttZ_2017_tautau = data_background_ttV + 'mc16d.410220.aMCPy8EG_tttautau.FS.nominal.root'
        ## 2015-2016
        self.input_path_background_ttZ_2016_nunu = data_background_ttV + 'mc16a.410156.aMCPy8EG_ttZnunu.FS.nominal.root'
        self.input_path_background_ttZ_2016_mumu = data_background_ttV + 'mc16a.410219.aMCPy8EG_ttmumu.FS.nominal.root'
        self.input_path_background_ttZ_2016_tautau = data_background_ttV + 'mc16a.410220.aMCPy8EG_tttautau.FS.nominal.root'

        ### Background ttbar
        ## 2018
        self.input_path_background_ttbar_l_2018 = data_background_ttbar + 'mc16e.410470.PhPy8EG_ttbar_hdamp258p75_l.FS.nominal.root'
        self.input_path_background_ttbar_0l_2018 = data_background_ttbar + 'mc16e.410471.PhPy8EG_ttbar_hdamp258p75_0l.FS.nominal.root'
        self.input_path_background_ttbar_2l_2018 = data_background_ttbar + 'mc16e.410472.PhPy8EG_ttbar_hdamp258p75_2l.FS.nominal.root'
        ## 2017
        self.input_path_background_ttbar_l_2017 = data_background_ttbar + 'mc16d.410470.PhPy8EG_ttbar_hdamp258p75_l.FS.nominal.root'
        self.input_path_background_ttbar_0l_2017 = data_background_ttbar + 'mc16d.410471.PhPy8EG_ttbar_hdamp258p75_0l.FS.nominal.root'
        self.input_path_background_ttbar_2l_2017 = data_background_ttbar + 'mc16d.410472.PhPy8EG_ttbar_hdamp258p75_2l.FS.nominal.root'        
        ## 2015-2016
        self.input_path_background_ttbar_l_2016 = data_background_ttbar + 'mc16a.410470.PhPy8EG_ttbar_hdamp258p75_l.FS.nominal.root'
        self.input_path_background_ttbar_0l_2016 = data_background_ttbar + 'mc16a.410471.PhPy8EG_ttbar_hdamp258p75_0l.FS.nominal.root'
        self.input_path_background_ttbar_2l_2016 = data_background_ttbar + 'mc16a.410472.PhPy8EG_ttbar_hdamp258p75_2l.FS.nominal.root'

        ## Background tWZ
        self.input_path_background_tWZ_2018 = data_background_tWZ + 'mc16e.410408.aMCPy8EG_tWZ_Ztoll_minDR1.FS.nominal.root'
        self.input_path_background_tWZ_2017 = data_background_tWZ + 'mc16d.410408.aMCPy8EG_tWZ_Ztoll_minDR1.FS.nominal.root'
        self.input_path_background_tWZ_2016 = data_background_tWZ + 'mc16a.410408.aMCPy8EG_tWZ_Ztoll_minDR1.FS.nominal.root'

        self.signal_sample = "tHqLoop_nominal;1"
        self.background_sample = "tHqLoop_nominal;1"
        ## Signal Tree tZq 2015-2018
        self.signal_tree_2018 = ur.open(self.input_path_sample_2018)[self.signal_sample]
        self.signal_tree_2017 = ur.open(self.input_path_sample_2017)[self.signal_sample]   
        self.signal_tree_15_16 = ur.open(self.input_path_sample_15_16)[self.signal_sample]     
        ## Background Tree diboson 2018
        self.background_tree_diboson_2018_ZqqZvv = ur.open(self.input_path_background_diboson_2018_ZqqZvv)[self.background_sample]
        self.background_tree_diboson_2018_ZqqZll = ur.open(self.input_path_background_diboson_2018_ZqqZll)[self.background_sample]
        self.background_tree_diboson_2018_WqqZvv = ur.open(self.input_path_background_diboson_2018_WqqZvv)[self.background_sample]
        self.background_tree_diboson_2018_WqqZll = ur.open(self.input_path_background_diboson_2018_WqqZll)[self.background_sample]
        self.background_tree_diboson_2018_WpqqWmlv = ur.open(self.input_path_background_diboson_2018_WpqqWmlv)[self.background_sample]
        self.background_tree_diboson_2018_WplvWmqq = ur.open(self.input_path_background_diboson_2018_WplvWmqq)[self.background_sample]
        self.background_tree_diboson_2018_WlvZqq = ur.open(self.input_path_background_diboson_2018_WlvZqq)[self.background_sample]
        self.background_tree_diboson_2018_llll = ur.open(self.input_path_background_diboson_2018_llll)[self.background_sample]
        self.background_tree_diboson_2018_lllv = ur.open(self.input_path_background_diboson_2018_lllv)[self.background_sample]
        self.background_tree_diboson_2018_llvv = ur.open(self.input_path_background_diboson_2018_llvv)[self.background_sample]
        self.background_tree_diboson_2018_lvvv = ur.open(self.input_path_background_diboson_2018_lvvv)[self.background_sample]
        self.background_tree_diboson_2018_llll_lowMll = ur.open(self.input_path_background_diboson_2018_llll_lowMll)[self.background_sample]
        self.background_tree_diboson_2018_lllv_lowMll = ur.open(self.input_path_background_diboson_2018_lllv_lowMll)[self.background_sample]
        self.background_tree_diboson_2018_llvv_lowMll = ur.open(self.input_path_background_diboson_2018_llvv_lowMll)[self.background_sample]
        ## Background Tree diboson 2017
        self.background_tree_diboson_2017_ZqqZll = ur.open(self.input_path_background_diboson_2017_ZqqZll)[self.background_sample]
        self.background_tree_diboson_2017_WqqZvv = ur.open(self.input_path_background_diboson_2017_WqqZvv)[self.background_sample]
        self.background_tree_diboson_2017_WqqZll = ur.open(self.input_path_background_diboson_2017_WqqZll)[self.background_sample]
        self.background_tree_diboson_2017_WpqqWmlv = ur.open(self.input_path_background_diboson_2017_WpqqWmlv)[self.background_sample]
        self.background_tree_diboson_2017_WplvWmqq = ur.open(self.input_path_background_diboson_2017_WplvWmqq)[self.background_sample]
        self.background_tree_diboson_2017_WlvZqq = ur.open(self.input_path_background_diboson_2017_WlvZqq)[self.background_sample]
        self.background_tree_diboson_2017_llll = ur.open(self.input_path_background_diboson_2017_llll)[self.background_sample]
        self.background_tree_diboson_2017_lllv = ur.open(self.input_path_background_diboson_2017_lllv)[self.background_sample]
        self.background_tree_diboson_2017_llvv = ur.open(self.input_path_background_diboson_2017_llvv)[self.background_sample]
        self.background_tree_diboson_2017_lvvv = ur.open(self.input_path_background_diboson_2017_lvvv)[self.background_sample]
        self.background_tree_diboson_2017_llll_lowMll = ur.open(self.input_path_background_diboson_2017_llll_lowMll)[self.background_sample]
        self.background_tree_diboson_2017_lllv_lowMll = ur.open(self.input_path_background_diboson_2017_lllv_lowMll)[self.background_sample]
        self.background_tree_diboson_2017_llvv_lowMll = ur.open(self.input_path_background_diboson_2017_llvv_lowMll)[self.background_sample]
        ## Background Tree diboson 2015-2016
        self.background_tree_diboson_2016_ZqqZvv = ur.open(self.input_path_background_diboson_2016_ZqqZvv)[self.background_sample]
        self.background_tree_diboson_2016_ZqqZll = ur.open(self.input_path_background_diboson_2016_ZqqZll)[self.background_sample]
        self.background_tree_diboson_2016_WqqZvv = ur.open(self.input_path_background_diboson_2016_WqqZvv)[self.background_sample]
        self.background_tree_diboson_2016_WqqZll = ur.open(self.input_path_background_diboson_2016_WqqZll)[self.background_sample]
        self.background_tree_diboson_2016_WpqqWmlv = ur.open(self.input_path_background_diboson_2016_WpqqWmlv)[self.background_sample]
        self.background_tree_diboson_2016_WplvWmqq = ur.open(self.input_path_background_diboson_2016_WplvWmqq)[self.background_sample]
        self.background_tree_diboson_2016_WlvZqq = ur.open(self.input_path_background_diboson_2016_WlvZqq)[self.background_sample]
        self.background_tree_diboson_2016_llll = ur.open(self.input_path_background_diboson_2016_llll)[self.background_sample]
        self.background_tree_diboson_2016_lllv = ur.open(self.input_path_background_diboson_2016_lllv)[self.background_sample]
        self.background_tree_diboson_2016_llvv = ur.open(self.input_path_background_diboson_2016_llvv)[self.background_sample]
        self.background_tree_diboson_2016_lvvv = ur.open(self.input_path_background_diboson_2016_lvvv)[self.background_sample]
        self.background_tree_diboson_2016_llll_lowMll = ur.open(self.input_path_background_diboson_2016_llll_lowMll)[self.background_sample]
        self.background_tree_diboson_2016_lllv_lowMll = ur.open(self.input_path_background_diboson_2016_lllv_lowMll)[self.background_sample]
        self.background_tree_diboson_2016_llvv_lowMll = ur.open(self.input_path_background_diboson_2016_llvv_lowMll)[self.background_sample]
        ## Background Tree ttZ 2018
        self.background_tree_ttZ_2018_ee = ur.open(self.input_path_background_ttZ_2018_ee)[self.background_sample]
        self.background_tree_ttZ_2018_nunu = ur.open(self.input_path_background_ttZ_2018_nunu)[self.background_sample]
        self.background_tree_ttZ_2018_mumu = ur.open(self.input_path_background_ttZ_2018_mumu)[self.background_sample]
        self.background_tree_ttZ_2018_tautau = ur.open(self.input_path_background_ttZ_2018_tautau)[self.background_sample]
        ## Background Tree ttZ 2017
        self.background_tree_ttZ_2017_ee = ur.open(self.input_path_background_ttZ_2017_ee)[self.background_sample]
        self.background_tree_ttZ_2017_nunu = ur.open(self.input_path_background_ttZ_2017_nunu)[self.background_sample]
        self.background_tree_ttZ_2017_mumu = ur.open(self.input_path_background_ttZ_2017_mumu)[self.background_sample]
        self.background_tree_ttZ_2017_tautau = ur.open(self.input_path_background_ttZ_2017_tautau)[self.background_sample]
        ## Background Tree ttZ 2015-2016
        self.background_tree_ttZ_2016_ee = ur.open(self.input_path_background_ttZ_2016_ee)[self.background_sample]
        self.background_tree_ttZ_2016_nunu = ur.open(self.input_path_background_ttZ_2016_nunu)[self.background_sample]
        self.background_tree_ttZ_2016_mumu = ur.open(self.input_path_background_ttZ_2016_mumu)[self.background_sample]
        self.background_tree_ttZ_2016_tautau = ur.open(self.input_path_background_ttZ_2016_tautau)[self.background_sample]
        ## Background Tree ttbar 2018
        self.background_tree_ttbar_l_2018 = ur.open(self.input_path_background_ttbar_l_2018)[self.background_sample]
        self.background_tree_ttbar_0l_2018 = ur.open(self.input_path_background_ttbar_0l_2018)[self.background_sample]
        self.background_tree_ttbar_2l_2018 = ur.open(self.input_path_background_ttbar_2l_2018)[self.background_sample]
        ## Background Tree ttbar 2017
        self.background_tree_ttbar_l_2017 = ur.open(self.input_path_background_ttbar_l_2017)[self.background_sample]
        self.background_tree_ttbar_0l_2017 = ur.open(self.input_path_background_ttbar_0l_2017)[self.background_sample]
        self.background_tree_ttbar_2l_2017 = ur.open(self.input_path_background_ttbar_2l_2017)[self.background_sample]
        ## Background Tree ttbar 2015-2016
        self.background_tree_ttbar_l_2016 = ur.open(self.input_path_background_ttbar_l_2016)[self.background_sample]
        self.background_tree_ttbar_0l_2016 = ur.open(self.input_path_background_ttbar_0l_2016)[self.background_sample]
        self.background_tree_ttbar_2l_2016 = ur.open(self.input_path_background_ttbar_2l_2016)[self.background_sample]
        ## Background Tree tWZ 2018 
        self.background_tree_tWZ_2018 = ur.open(self.input_path_background_tWZ_2018)[self.background_sample]
        ## Background Tree tWZ 2017
        self.background_tree_tWZ_2017 = ur.open(self.input_path_background_tWZ_2017)[self.background_sample]
        ## Background Tree tWZ 2015-2016 
        self.background_tree_tWZ_2016 = ur.open(self.input_path_background_tWZ_2016)[self.background_sample]

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
        self.discriminator_epochs = 250
        self.batchSize = 512
        #Setup of the networks, nodes and layers
        self.discriminator_layers = 3
        self.discriminator_nodes = 128
        #Setup of the networks, loss and optimisation
        ## just an integer
        self.queue = 3
        ##
        self.my_optimizer = 'SGD'
        self.discriminator_lr = float(sys.argv[1])
        self.discriminator_momentum = 0.9
        self.discriminator_optimizer = SGD(lr = self.discriminator_lr, momentum = self.discriminator_momentum)
        self.discriminator_optimizer_adam = Adam(lr = self.discriminator_lr,beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay = self.discriminator_lr/self.discriminator_epochs)
        self.discriminator_dropout = 0.3
        self.discriminator_loss = 'binary_crossentropy'
        self.validation_fraction = 0.1

        self.output_job = output_path + 'epochs_%i/lr_%.6f/momentum_%.3f/' % (self.discriminator_epochs,self.discriminator_lr,self.discriminator_momentum)
        self.output_lr = output_path + 'epochs_%i/' % (self.discriminator_epochs)
        self.output_curve = self.output_lr + 'txtlr/'
        if not os.path.exists(self.output_job):
            os.makedirs(self.output_job)
        if not os.path.exists(self.output_curve):
            os.makedirs(self.output_curve)


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

        ### Numpy conversion of Signal tZq 2015-2018
        self.events_signal_2018 = self.signal_tree_2018.pandas.df(self.variables).to_numpy()
        self.events_signal_2017 = self.signal_tree_2017.pandas.df(self.variables).to_numpy()
        self.events_signal_15_16 = self.signal_tree_15_16.pandas.df(self.variables).to_numpy()

        self.events_signal = np.concatenate([self.events_signal_2018,self.events_signal_2017,self.events_signal_15_16])


        ### Numpy conversion of diboson background 
        ## 2018
        self.events_background_diboson_2018_ZqqZvv = self.background_tree_diboson_2018_ZqqZvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_ZqqZll = self.background_tree_diboson_2018_ZqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_WqqZvv = self.background_tree_diboson_2018_WqqZvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_WqqZll = self.background_tree_diboson_2018_WqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_WpqqWmlv = self.background_tree_diboson_2018_WpqqWmlv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_WplvWmqq = self.background_tree_diboson_2018_WplvWmqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_WlvZqq = self.background_tree_diboson_2018_WlvZqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_llll = self.background_tree_diboson_2018_llll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_lllv = self.background_tree_diboson_2018_lllv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_llvv = self.background_tree_diboson_2018_llvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_lvvv = self.background_tree_diboson_2018_lvvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_llll_lowMll = self.background_tree_diboson_2018_llll_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_lllv_lowMll = self.background_tree_diboson_2018_lllv_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2018_llvv_lowMll = self.background_tree_diboson_2018_llvv_lowMll.pandas.df(self.variables).to_numpy()
    
        self.events_background_diboson_2018 = np.concatenate([self.events_background_diboson_2018_ZqqZvv,self.events_background_diboson_2018_ZqqZll,self.events_background_diboson_2018_WqqZvv,self.events_background_diboson_2018_WqqZll,self.events_background_diboson_2018_WpqqWmlv,self.events_background_diboson_2018_WplvWmqq,self.events_background_diboson_2018_WlvZqq,self.events_background_diboson_2018_llll,self.events_background_diboson_2018_lllv,self.events_background_diboson_2018_llvv,self.events_background_diboson_2018_lvvv,self.events_background_diboson_2018_llll_lowMll,self.events_background_diboson_2018_lllv_lowMll,self.events_background_diboson_2018_llvv_lowMll])
        ## 2017
        self.events_background_diboson_2017_ZqqZll = self.background_tree_diboson_2017_ZqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_WqqZvv = self.background_tree_diboson_2017_WqqZvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_WqqZll = self.background_tree_diboson_2017_WqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_WpqqWmlv = self.background_tree_diboson_2017_WpqqWmlv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_WplvWmqq = self.background_tree_diboson_2017_WplvWmqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_WlvZqq = self.background_tree_diboson_2017_WlvZqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_llll = self.background_tree_diboson_2017_llll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_lllv = self.background_tree_diboson_2017_lllv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_llvv = self.background_tree_diboson_2017_llvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_lvvv = self.background_tree_diboson_2017_lvvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_llll_lowMll = self.background_tree_diboson_2017_llll_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_lllv_lowMll = self.background_tree_diboson_2017_lllv_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2017_llvv_lowMll = self.background_tree_diboson_2017_llvv_lowMll.pandas.df(self.variables).to_numpy()

        self.events_background_diboson_2017 = np.concatenate([self.events_background_diboson_2017_ZqqZll,self.events_background_diboson_2017_WqqZvv,self.events_background_diboson_2017_WqqZll,self.events_background_diboson_2017_WpqqWmlv,self.events_background_diboson_2017_WplvWmqq,self.events_background_diboson_2017_WlvZqq,self.events_background_diboson_2017_llll,self.events_background_diboson_2017_lllv,self.events_background_diboson_2017_llvv,self.events_background_diboson_2017_lvvv,self.events_background_diboson_2017_llll_lowMll,self.events_background_diboson_2017_lllv_lowMll,self.events_background_diboson_2017_llvv_lowMll])

        ##2015-2016
        self.events_background_diboson_2016_ZqqZvv = self.background_tree_diboson_2016_ZqqZvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_ZqqZll = self.background_tree_diboson_2016_ZqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_WqqZvv = self.background_tree_diboson_2016_WqqZvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_WqqZll = self.background_tree_diboson_2016_WqqZll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_WpqqWmlv = self.background_tree_diboson_2016_WpqqWmlv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_WplvWmqq = self.background_tree_diboson_2016_WplvWmqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_WlvZqq = self.background_tree_diboson_2016_WlvZqq.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_llll = self.background_tree_diboson_2016_llll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_lllv = self.background_tree_diboson_2016_lllv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_llvv = self.background_tree_diboson_2016_llvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_lvvv = self.background_tree_diboson_2016_lvvv.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_llll_lowMll = self.background_tree_diboson_2016_llll_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_lllv_lowMll = self.background_tree_diboson_2016_lllv_lowMll.pandas.df(self.variables).to_numpy()
        self.events_background_diboson_2016_llvv_lowMll = self.background_tree_diboson_2016_llvv_lowMll.pandas.df(self.variables).to_numpy()        

        self.events_background_diboson_2016 = np.concatenate([self.events_background_diboson_2016_ZqqZvv,self.events_background_diboson_2016_ZqqZll,self.events_background_diboson_2016_WqqZvv,self.events_background_diboson_2016_WqqZll,self.events_background_diboson_2016_WpqqWmlv,self.events_background_diboson_2016_WplvWmqq,self.events_background_diboson_2016_WlvZqq,self.events_background_diboson_2016_llll,self.events_background_diboson_2016_lllv,self.events_background_diboson_2016_llvv,self.events_background_diboson_2016_lvvv,self.events_background_diboson_2016_llll_lowMll,self.events_background_diboson_2016_lllv_lowMll,self.events_background_diboson_2016_llvv_lowMll])


        ## Background Diboson put together 
        self.events_background_diboson = np.concatenate([self.events_background_diboson_2018,self.events_background_diboson_2017,self.events_background_diboson_2016])

        ### Numpy conversion of ttZ background 
        ## 2018
        self.events_background_ttZ_2018_ee = self.background_tree_ttZ_2018_ee.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2018_nunu = self.background_tree_ttZ_2018_nunu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2018_mumu = self.background_tree_ttZ_2018_mumu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2018_tautau = self.background_tree_ttZ_2018_tautau.pandas.df(self.variables).to_numpy()

        self.events_background_ttZ_2018 = np.concatenate([self.events_background_ttZ_2018_ee,self.events_background_ttZ_2018_nunu,self.events_background_ttZ_2018_mumu,self.events_background_ttZ_2018_tautau])
        ## 2017
        self.events_background_ttZ_2017_ee = self.background_tree_ttZ_2017_ee.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2017_nunu = self.background_tree_ttZ_2017_nunu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2017_mumu = self.background_tree_ttZ_2017_mumu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2017_tautau = self.background_tree_ttZ_2017_tautau.pandas.df(self.variables).to_numpy()

        self.events_background_ttZ_2017 = np.concatenate([self.events_background_ttZ_2017_ee,self.events_background_ttZ_2017_nunu,self.events_background_ttZ_2017_mumu,self.events_background_ttZ_2017_tautau])
        ## 2015-2016
        self.events_background_ttZ_2016_ee = self.background_tree_ttZ_2016_ee.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2016_nunu = self.background_tree_ttZ_2016_nunu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2016_mumu = self.background_tree_ttZ_2016_mumu.pandas.df(self.variables).to_numpy()
        self.events_background_ttZ_2016_tautau = self.background_tree_ttZ_2016_tautau.pandas.df(self.variables).to_numpy()

        self.events_background_ttZ_2016 = np.concatenate([self.events_background_ttZ_2016_ee,self.events_background_ttZ_2016_nunu,self.events_background_ttZ_2016_mumu,self.events_background_ttZ_2016_tautau])       
        ## ttZ background put together 
        self.events_background_ttZ = np.concatenate([self.events_background_ttZ_2018,self.events_background_ttZ_2017,self.events_background_ttZ_2016])

        ### Numpy conversion of tWZ background
        ## 2018
        self.events_background_tWZ_2018 = self.background_tree_tWZ_2018.pandas.df(self.variables).to_numpy()
        ## 2017
        self.events_background_tWZ_2017 = self.background_tree_tWZ_2017.pandas.df(self.variables).to_numpy()
        ## 2016
        self.events_background_tWZ_2016 = self.background_tree_tWZ_2016.pandas.df(self.variables).to_numpy()

        self.events_background_tWZ = np.concatenate([self.events_background_tWZ_2018,self.events_background_tWZ_2017,self.events_background_tWZ_2016])

        ### Numpy conversion of ttbar background
        ## 2018
        self.events_background_ttbar_l_2018 = self.background_tree_ttbar_l_2018.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_0l_2018 = self.background_tree_ttbar_0l_2018.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_2018 = np.concatenate([self.events_background_ttbar_l_2018,self.events_background_ttbar_0l_2018])
        ## 2017
        self.events_background_ttbar_l_2017 = self.background_tree_ttbar_l_2017.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_0l_2017 = self.background_tree_ttbar_0l_2017.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_2017 = np.concatenate([self.events_background_ttbar_l_2017,self.events_background_ttbar_0l_2017])
        ## 2016
        self.events_background_ttbar_l_2016 = self.background_tree_ttbar_l_2016.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_0l_2016 = self.background_tree_ttbar_0l_2016.pandas.df(self.variables).to_numpy()
        self.events_background_ttbar_2016 = np.concatenate([self.events_background_ttbar_l_2016,self.events_background_ttbar_0l_2016])
        ## Background ttbar put together 
        self.events_background_ttbar = np.concatenate([self.events_background_ttbar_2018,self.events_background_ttbar_2017,self.events_background_ttbar_2016])

        self.events_background_ttZ_tWZ = np.concatenate([self.events_background_tWZ, self.events_background_ttZ])

        ##All backgrounds put together (Care for same order in weights)
        self.events_background = np.concatenate([self.events_background_diboson,self.events_background_ttZ,self.events_background_ttbar,self.events_background_tWZ])

        #Setting up the weights. The weights for each tree are stored in 'weight_nominal'
        self.weight_signal_2018 = self.signal_tree_2018.pandas.df('weight_nominal').to_numpy()
        self.weight_signal_2017 = self.signal_tree_2017.pandas.df('weight_nominal').to_numpy()
        self.weight_signal_15_16 = self.signal_tree_15_16.pandas.df('weight_nominal').to_numpy()

        self.weight_signal = np.concatenate([self.weight_signal_2018,self.weight_signal_2017,self.weight_signal_15_16])
        print(self.weight_signal.sum()*139)
        self.weight_signal = np.absolute(self.weight_signal) * 139
        ### Numpy conversion of diboson weights
        ## 2018
        self.weight_background_diboson_2018_ZqqZvv = self.background_tree_diboson_2018_ZqqZvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_ZqqZll = self.background_tree_diboson_2018_ZqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_WqqZvv = self.background_tree_diboson_2018_WqqZvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_WqqZll = self.background_tree_diboson_2018_WqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_WpqqWmlv = self.background_tree_diboson_2018_WpqqWmlv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_WplvWmqq = self.background_tree_diboson_2018_WplvWmqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_WlvZqq = self.background_tree_diboson_2018_WlvZqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_llll = self.background_tree_diboson_2018_llll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_lllv = self.background_tree_diboson_2018_lllv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_llvv = self.background_tree_diboson_2018_llvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_lvvv = self.background_tree_diboson_2018_lvvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_llll_lowMll = self.background_tree_diboson_2018_llll_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_lllv_lowMll = self.background_tree_diboson_2018_lllv_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2018_llvv_lowMll = self.background_tree_diboson_2018_llvv_lowMll.pandas.df('weight_nominal').to_numpy()

        self.weight_background_diboson_2018 = np.concatenate([self.weight_background_diboson_2018_ZqqZvv,self.weight_background_diboson_2018_ZqqZll,self.weight_background_diboson_2018_WqqZvv,self.weight_background_diboson_2018_WqqZll,self.weight_background_diboson_2018_WpqqWmlv,self.weight_background_diboson_2018_WplvWmqq,self.weight_background_diboson_2018_WlvZqq,self.weight_background_diboson_2018_llll,self.weight_background_diboson_2018_lllv,self.weight_background_diboson_2018_llvv,self.weight_background_diboson_2018_lvvv,self.weight_background_diboson_2018_llll_lowMll,self.weight_background_diboson_2018_lllv_lowMll,self.weight_background_diboson_2018_llvv_lowMll])
        ## 2017 
        self.weight_background_diboson_2017_ZqqZll = self.background_tree_diboson_2017_ZqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_WqqZvv = self.background_tree_diboson_2017_WqqZvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_WqqZll = self.background_tree_diboson_2017_WqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_WpqqWmlv = self.background_tree_diboson_2017_WpqqWmlv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_WplvWmqq = self.background_tree_diboson_2017_WplvWmqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_WlvZqq = self.background_tree_diboson_2017_WlvZqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_llll = self.background_tree_diboson_2017_llll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_lllv = self.background_tree_diboson_2017_lllv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_llvv = self.background_tree_diboson_2017_llvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_lvvv = self.background_tree_diboson_2017_lvvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_llll_lowMll = self.background_tree_diboson_2017_llll_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_lllv_lowMll = self.background_tree_diboson_2017_lllv_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2017_llvv_lowMll = self.background_tree_diboson_2017_llvv_lowMll.pandas.df('weight_nominal').to_numpy()

        self.weight_background_diboson_2017 = np.concatenate([self.weight_background_diboson_2017_ZqqZll,self.weight_background_diboson_2017_WqqZvv,self.weight_background_diboson_2017_WqqZll,self.weight_background_diboson_2017_WpqqWmlv,self.weight_background_diboson_2017_WplvWmqq,self.weight_background_diboson_2017_WlvZqq,self.weight_background_diboson_2017_llll,self.weight_background_diboson_2017_lllv,self.weight_background_diboson_2017_llvv,self.weight_background_diboson_2017_lvvv,self.weight_background_diboson_2017_llll_lowMll,self.weight_background_diboson_2017_lllv_lowMll,self.weight_background_diboson_2017_llvv_lowMll])

        ## 2016
        self.weight_background_diboson_2016_ZqqZvv = self.background_tree_diboson_2016_ZqqZvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_ZqqZll = self.background_tree_diboson_2016_ZqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_WqqZvv = self.background_tree_diboson_2016_WqqZvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_WqqZll = self.background_tree_diboson_2016_WqqZll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_WpqqWmlv = self.background_tree_diboson_2016_WpqqWmlv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_WplvWmqq = self.background_tree_diboson_2016_WplvWmqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_WlvZqq = self.background_tree_diboson_2016_WlvZqq.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_llll = self.background_tree_diboson_2016_llll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_lllv = self.background_tree_diboson_2016_lllv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_llvv = self.background_tree_diboson_2016_llvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_lvvv = self.background_tree_diboson_2016_lvvv.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_llll_lowMll = self.background_tree_diboson_2016_llll_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_lllv_lowMll = self.background_tree_diboson_2016_lllv_lowMll.pandas.df('weight_nominal').to_numpy()
        self.weight_background_diboson_2016_llvv_lowMll = self.background_tree_diboson_2016_llvv_lowMll.pandas.df('weight_nominal').to_numpy()

        self.weight_background_diboson_2016 = np.concatenate([self.weight_background_diboson_2016_ZqqZvv,self.weight_background_diboson_2016_ZqqZll,self.weight_background_diboson_2016_WqqZvv,self.weight_background_diboson_2016_WqqZll,self.weight_background_diboson_2016_WpqqWmlv,self.weight_background_diboson_2016_WplvWmqq,self.weight_background_diboson_2016_WlvZqq,self.weight_background_diboson_2016_llll,self.weight_background_diboson_2016_lllv,self.weight_background_diboson_2016_llvv,self.weight_background_diboson_2016_lvvv,self.weight_background_diboson_2016_llll_lowMll,self.weight_background_diboson_2016_lllv_lowMll,self.weight_background_diboson_2016_llvv_lowMll])

        ### diboson weight background put together
        self.weight_background_diboson = np.concatenate([self.weight_background_diboson_2018,self.weight_background_diboson_2017,self.weight_background_diboson_2016])

        ### Numpy conversion of weights background ttZ
        ## 2018
        self.weight_background_ttZ_2018_ee = self.background_tree_ttZ_2018_ee.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2018_nunu = self.background_tree_ttZ_2018_nunu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2018_mumu = self.background_tree_ttZ_2018_mumu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2018_tautau = self.background_tree_ttZ_2018_tautau.pandas.df('weight_nominal').to_numpy()

        self.weight_background_ttZ_2018 = np.concatenate([self.weight_background_ttZ_2018_ee,self.weight_background_ttZ_2018_nunu,self.weight_background_ttZ_2018_mumu,self.weight_background_ttZ_2018_tautau])
        ## 2017
        self.weight_background_ttZ_2017_ee = self.background_tree_ttZ_2017_ee.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2017_nunu = self.background_tree_ttZ_2017_nunu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2017_mumu = self.background_tree_ttZ_2017_mumu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2017_tautau = self.background_tree_ttZ_2017_tautau.pandas.df('weight_nominal').to_numpy()

        self.weight_background_ttZ_2017 = np.concatenate([self.weight_background_ttZ_2017_ee,self.weight_background_ttZ_2017_nunu,self.weight_background_ttZ_2017_mumu,self.weight_background_ttZ_2017_tautau])
        ## 2016
        self.weight_background_ttZ_2016_ee = self.background_tree_ttZ_2016_ee.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2016_nunu = self.background_tree_ttZ_2016_nunu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2016_mumu = self.background_tree_ttZ_2016_mumu.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttZ_2016_tautau = self.background_tree_ttZ_2016_tautau.pandas.df('weight_nominal').to_numpy()

        self.weight_background_ttZ_2016 = np.concatenate([self.weight_background_ttZ_2016_ee,self.weight_background_ttZ_2016_nunu,self.weight_background_ttZ_2016_mumu,self.weight_background_ttZ_2016_tautau])
        ## ttZ background put together 
        self.weight_background_ttZ = np.concatenate([self.weight_background_ttZ_2018,self.weight_background_ttZ_2017,self.weight_background_ttZ_2016])
        
        ### Numpy conversion of weights background tWZ
        ## 2018
        self.weight_background_tWZ_2018 = self.background_tree_tWZ_2018.pandas.df('weight_nominal').to_numpy()
        ## 2017
        self.weight_background_tWZ_2017 = self.background_tree_tWZ_2017.pandas.df('weight_nominal').to_numpy()
        ## 2016
        self.weight_background_tWZ_2016 = self.background_tree_tWZ_2016.pandas.df('weight_nominal').to_numpy()

        # tWZ background put together 
        self.weight_background_tWZ = np.concatenate([self.weight_background_tWZ_2018,self.weight_background_tWZ_2017,self.weight_background_tWZ_2016])

        ### Numpy conversion of weights background ttbar
        ## 2018
        self.weight_background_ttbar_l_2018 = self.background_tree_ttbar_l_2018.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_0l_2018 = self.background_tree_ttbar_0l_2018.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_2018 = np.concatenate([self.weight_background_ttbar_l_2018,self.weight_background_ttbar_0l_2018])
        ## 2017
        self.weight_background_ttbar_l_2017 = self.background_tree_ttbar_l_2017.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_0l_2017 = self.background_tree_ttbar_0l_2017.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_2017 = np.concatenate([self.weight_background_ttbar_l_2017,self.weight_background_ttbar_0l_2017])
        ## 2015-2016
        self.weight_background_ttbar_l_2016 = self.background_tree_ttbar_l_2016.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_0l_2016 = self.background_tree_ttbar_0l_2016.pandas.df('weight_nominal').to_numpy()
        self.weight_background_ttbar_2016 = np.concatenate([self.weight_background_ttbar_l_2016,self.weight_background_ttbar_0l_2016])
        ### ttbar weight background put together
        self.weight_background_ttbar = np.concatenate([self.weight_background_ttbar_2018,self.weight_background_ttbar_2017,self.weight_background_ttbar_2016])
        ## Weights put together in same order as background
        self.weight_background = np.concatenate([self.weight_background_diboson,self.weight_background_ttZ,self.weight_background_ttbar,self.weight_background_tWZ])
        print(self.weight_background_diboson.sum()*140)
        print(self.weight_background_ttZ.sum()*140)
        print(self.weight_background_ttbar.sum()*140)
        self.weight_background = np.absolute(self.weight_background) * 139

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
        #The samples and corresponding targets are ZqqZlnow split into a sample for training and a sample for testing. Keep in mind that the same random seed should be used for both splits
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
        self.model.compile(loss=binary_crossentropy,weighted_metrics = [metrics.binary_accuracy],optimizer = self.discriminator_optimizer_adam)
        #self.model.compile(loss='binary_crossentropy',weighted_metrics =[metrics.binary_accuracy],optimizer = self.discriminator_optimizer)
        #self.model.summary()
        

    def trainDiscriminator(self):

        #print(self.target_training[12:500])
        #print(self.target_training[-1:-100])Tipp: Wenn Sie sich mit einem Google-Konto anmelden, bevor Sie zustimmen, wird Ihre Auswahl auf allen Geräten und in allen Browsern gespeichert, bei denen Sie angemeldet sind.

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
        plt.plot(self.fpr_test, self.tpr_test, 'g--',color ='lime', label='$AUC_{test}$ = %0.2f'% self.auc_test)
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
                self.signal_histo.append(self.model_prediction_test[i])
            if self.target_validation[i] == 0:
                self.background_histo.append(self.model_prediction_test[i])
                
        plt.hist(self.signal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step",density = True,color=color_tW, label = "Signal")
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

        file = open(self.output_curve + 'lrcurve_%.6f.txt'%(self.discriminator_lr),'w')
        file.write('%.4e,%.4e,%.4e,%.4e'%(self.discriminator_history.history['loss'][-1],self.discriminator_history.history['val_loss'][-1],self.discriminator_lr,self.auc))
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
        plt.plot(lr_list,loss_plot,color = color_tt,label='Training',marker = 'x',linestyle = 'None')
        plt.plot(lr_list,val_loss_plot,color = color_tW,label = 'Test',marker = 'x',linestyle = 'None')
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-1,2),useMathText = True)
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Loss Curve for %i Epochs' % (self.discriminator_epochs))
        plt.legend()
        plt.gcf().savefig(self.output_lr+'LRPlot.png')
        plt.gcf().clear()
        plt.plot(lr_list,auc_list, color = 'navy', marker = 'x', linestyle = 'None')
        plt.xlabel('Learning rate')
        plt.ylabel('Auc-Value')
        plt.title('Learning Rate Auc Curve')
        plt.gcf().savefig(self.output_lr+'LRAucPlot.png')
        plt.gcf().clear()

        ###
            
        



    def HistObject(self,Xaxisbins,Yaxisbins,range1,range2,bins,labelxaxis,savelabel,numbervariable):
        self.hist_tZq = self.events_signal.transpose() 
        self.hist_diboson = self.events_background_diboson.transpose() 
        self.hist_ttbar = self.events_background_ttbar.transpose()
        self.hist_ttZ_tWZ = self.events_background_ttZ_tWZ.transpose()

        self.hist_tZq = np.clip(self.hist_tZq,None,range2)
        self.hist_diboson = np.clip(self.hist_diboson,None,range2)
        self.hist_ttbar = np.clip(self.hist_ttbar,None,range2)
        self.hist_ttZ_tWZ = np.clip(self.hist_ttZ_tWZ,None,range2)

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
        plt.hist(self.hist_ttbar[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='red',label=r'$t\bar{t}$',density = True)
        plt.hist(self.hist_ttZ_tWZ[numbervariable], range=[range1, range2], linewidth = .75, bins=bins, histtype="step", color='lime',label=r'$ttZ+tWZ$',density = True)

        plt.legend(frameon = False)

        plt.xlim(range1,range2)

        plt.xlabel(labelxaxis,horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)

        plt.gcf().savefig(mysavedata + savelabel +'.png')
        plt.gcf().clear()

    def Runtime(self,start,stop):
        file = open(self.output_job + 'params.txt','a')
        file.write('Runtime of program: %.2f seconds' % (stop-start))
        file.close()



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

###
#self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','m_met','m_top'])
#first_training.HistObject(10,5,0,800,16,'$m(bj_F)$','m_b_jf',0)
#first_training.HistObject(10,5,0,5,15,'$\eta(j_f)$','eta_jf',1)
#first_training.HistObject(5,10,-2.5,2.5,5,'$q(l^W)$','q_lW',2)
#first_training.HistObject(6,3,0,3.,12,'$\eta(l^W)$','eta_lW',3)
#first_training.HistObject(6,6,0,300.,11,'$p_T(W)$[GeV]','pT_W',4)
#first_training.HistObject(10,4,0,200.,11,'$p_T(l^W)$[GeV]','pT_lW',5)
#first_training.HistObject(6,5,60.,120.,25,'$m(ll)$[GeV]','m_Z',6)
#first_training.HistObject(10,4,0,5.,11,'$\eta(Z)$','eta_Z',7)
#first_training.HistObject(7,5,0,7.,16,'$\Delta R(j_f,Z)$','dR_jf_Z',8)
#first_training.HistObject(6,5,0,300.,11,'$p_T(j_f)$[GeV]','pT_jf',9)
#first_training.HistObject(10,6,0,200.,11,'$p_T(j_r)$[GeV]','pT_jr',10)
#first_training.HistObject(10,6,0,5.,16,'$\eta(j_r)$','eta_jr',11)
#first_training.HistObject(6,6,0,300.,11,'$p_T(Z)$[GeV]','pT_Z',12)
#first_training.HistObject(6,6,0,600,31,'$E^{miss}$[GeV]','m_met',13)
#first_training.HistObject(6,6,0,600,31,'$m_t$[GeV]','m_top',14)
###


### For LR Plot
text_files = [f for f in os.listdir(first_training.output_curve) if f.endswith('.txt')]
first_training.plot_lr(text_files)


###
end = timer()
first_training.Runtime(start,end)

