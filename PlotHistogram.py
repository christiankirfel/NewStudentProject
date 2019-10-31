import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uproot as ur
import pandas 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)


my_path_to_data = '/cephfs/user/s6pinogg/PietBachelor/18-10_tZVar/'
data_tzq = my_path_to_data + 'tZq/'
data_background_diboson = my_path_to_data + 'nBosons/'
mysavedata = '/cephfs/user/s6pinogg/PietBachelor/Histo_Data/'



if not os.path.exists(mysavedata):
    os.makedirs(mysavedata)

class PlotHisto(object):

    def __init__(self):
        self.variables = np.array(['m_b_jf','eta_jf','q_lW','eta_lW','pT_W','pT_lW','m_Z','eta_Z','dR_jf_Z','pT_jf','pT_jr','eta_jr','pT_Z','mT_W','m_top'])

        self.input_path_sample = data_tzq + "mc16a.412063.aMCPy8EG_tllq_nf4.FS.nominal.root"
        self.input_path_background = data_background_diboson + 'mc16a.364250.Sh222_llll.FS.nominal.root'

        self.signal_sample = "tHqLoop_nominal"
        self.background_sample = "tHqLoop_nominal"

        self.signal_tree = ur.open(self.input_path_sample)[self.signal_sample]
        self.background_tree = ur.open(self.input_path_background)[self.background_sample]

        self.signal_m_b_jf = self.signal_tree.pandas.df('m_b_jf').to_numpy()
        self.background_m_b_jf = self.background_tree.pandas.df('m_b_jf').to_numpy()

        self.signal_eta_jf = self.signal_tree.pandas.df('eta_jf').to_numpy()
        self.background_eta_jf = self.background_tree.pandas.df('eta_jf').to_numpy()

        self.signal_q_lW = self.signal_tree.pandas.df('q_lW').to_numpy()
        self.background_q_lW = self.background_tree.pandas.df('q_lW').to_numpy()

        self.signal_eta_lW = self.signal_tree.pandas.df('eta_lW').to_numpy()
        self.background_eta_lW = self.background_tree.pandas.df('eta_lW').to_numpy()

        self.signal_pT_W = self.signal_tree.pandas.df('pT_W').to_numpy()
        self.background_pT_W = self.background_tree.pandas.df('pT_W').to_numpy()

        self.signal_pT_lW = self.signal_tree.pandas.df('pT_lW').to_numpy()
        self.background_pT_lW = self.background_tree.pandas.df('pT_lW').to_numpy()


        self.signal_m_Z = self.signal_tree.pandas.df('m_Z').to_numpy()
        self.background_m_Z = self.background_tree.pandas.df('m_Z').to_numpy()


        self.signal_eta_Z = self.signal_tree.pandas.df('eta_Z').to_numpy()
        self.background_eta_Z = self.background_tree.pandas.df('eta_Z').to_numpy()

        self.signal_delR = self.signal_tree.pandas.df('dR_jf_Z').to_numpy()
        self.background_delR = self.background_tree.pandas.df('dR_jf_Z').to_numpy()

        self.signal_pT_jf = self.signal_tree.pandas.df('pT_jf').to_numpy()
        self.background_pT_jf = self.background_tree.pandas.df('pT_jf').to_numpy()  

        self.signal_pT_jr = self.signal_tree.pandas.df('pT_jr').to_numpy()
        self.background_pT_jr = self.background_tree.pandas.df('pT_jr').to_numpy()        

        self.signal_eta_jr = self.signal_tree.pandas.df('eta_jr').to_numpy()
        self.background_eta_jr = self.background_tree.pandas.df('eta_jr').to_numpy()               

        self.signal_pT_Z = self.signal_tree.pandas.df('pT_Z').to_numpy()
        self.background_pT_Z = self.background_tree.pandas.df('pT_Z').to_numpy() 

        self.signal_mT_W = self.signal_tree.pandas.df('mT_W').to_numpy()
        self.background_mT_W = self.background_tree.pandas.df('mT_W').to_numpy() 

        self.signal_m_top = self.signal_tree.pandas.df('m_top').to_numpy()
        self.background_m_top = self.background_tree.pandas.df('m_top').to_numpy() 


    def Plot_m_b_jf(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_m_b_jf, range=[0., 700.], linewidth = .5, bins=15, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_m_b_jf, range=[0., 700.], linewidth = .5, bins=15, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,700)

        plt.xlabel('$m(bj_f)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'm_b_jf.png')
        plt.gcf().clear()

    def Plot_eta_jf(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_eta_jf, range=[0., 5.], linewidth = .5, bins=15, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_eta_jf, range=[0., 5.], linewidth = .5, bins=15, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,5)

        plt.xlabel('$\eta(j_f)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'eta_jf.png')
        plt.gcf().clear()

    def Plot_q_lW(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4),useMathText=True)
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_q_lW, range=[-2.5, 2.5], linewidth = .5, bins=5, histtype="step", color='magenta',label='tZq',density = True)
        plt.legend(frameon = False)
        plt.xlim(-2.5,2.5)
        #plt.ylim(0,4000)

        plt.xlabel('$q(l^W)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'q_lW.png')
        plt.gcf().clear()

    def Plot_eta_lW(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_eta_lW, range=[0., 3], linewidth = .5, bins=12, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_eta_lW, range=[0., 3], linewidth = .5, bins=12, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,3)


        plt.xlabel('$\eta(l^W)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'eta_lW.png')
        plt.gcf().clear()        


    def Plot_pT_W(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_pT_W, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_pT_W, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,300)

        plt.xlabel('$p_T(W)$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'pT_W.png')
        plt.gcf().clear()    


    def Plot_pT_lW(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_pT_lW, range=[0., 200.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_pT_lW, range=[0., 200.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,200)

        plt.xlabel('$p_T(l^W)$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'pT_lW.png')
        plt.gcf().clear() 
   

    def Plot_m_Z(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_m_Z, range=[60., 120.], linewidth = .5, bins=24, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_m_Z, range=[60., 120.], linewidth = .5, bins=24, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(60,120)

        plt.xlabel('$m(ll)$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'm_Z.png')
        plt.gcf().clear() 

    def Plot_eta_Z(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_eta_Z, range=[0., 5.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_eta_Z, range=[0., 5.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,5)

        plt.xlabel('$\eta (Z)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'eta_Z.png')
        plt.gcf().clear() 

    def Plot_delR_jf_Z(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=7)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_delR, range=[0., 7.], linewidth = .5, bins=15, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_delR, range=[0., 7.], linewidth = .5, bins=15, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,7)

        plt.xlabel('$\Delta R(j_f,Z)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'dR_jf_Z.png')
        plt.gcf().clear() 

    def Plot_pT_jf(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_pT_jf, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_pT_jf, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,300)

        plt.xlabel('$p_T(j_f)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'pT_jf.png')
        plt.gcf().clear() 

    def Plot_pT_jr(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_pT_jr, range=[0., 200.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_pT_jr, range=[0., 200.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,200)

        plt.xlabel('$p_T(j_r)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'pT_jr.png')
        plt.gcf().clear() 

    def Plot_eta_jr(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_eta_jr, range=[0., 5.], linewidth = .5, bins=15, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_eta_jr, range=[0., 5.], linewidth = .5, bins=15, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,5)

        plt.xlabel('$\eta (j_r)$',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'eta_jr.png')
        plt.gcf().clear() 

    def Plot_pT_Z(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_pT_Z, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='magenta',label='tZq',density = True)
        plt.hist(self.background_pT_Z, range=[0., 300.], linewidth = .5, bins=10, histtype="step", color='royalblue', label ='DiBoson 4l',density = True)
        plt.legend(frameon = False)
        plt.xlim(0,300)

        plt.xlabel('$p_T(Z)$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Event density',va = 'top',y=0.87,labelpad=10)
        plt.gcf().savefig(mysavedata + 'pT_Z.png')
        plt.gcf().clear() 

'''
    def Plot_mT_W(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=12)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_mT_W, range=[0., 240.], linewidth = .5, bins=8, histtype="step", color='magenta',label='tZq')
        plt.hist(self.background_mT_W, range=[0., 240.], linewidth = .5, bins=8, histtype="step", color='royalblue', label ='DiBoson 4l')
        plt.legend(frameon = False)
        plt.xlim(0,240)

        plt.xlabel('$m_T(l,E_T^{miss})$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Events',va = 'top',y=0.95,labelpad=10)
        plt.gcf().savefig(mysavedata + 'mT_W.png')
        plt.gcf().clear()


    def Plot_m_top(self):
        ax = plt.subplot()
        ax.ticklabel_format(style='sci', axis ='both', scilimits=(-4,4))
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(direction='in',which='minor', length=2)
        plt.hist(self.signal_mT_W, range=[0., 600.], linewidth = .5, bins=30, histtype="step", color='magenta',label='tZq')
        plt.hist(self.background_mT_W, range=[0., 600.], linewidth = .5, bins=30, histtype="step", color='royalblue', label ='DiBoson 4l')
        plt.legend(frameon = False)
        plt.xlim(0,600)

        plt.xlabel('$m_t$[GeV]',horizontalalignment='right',x=1.0)
        plt.ylabel('Events',va = 'top',y=0.95,labelpad=10)
        plt.gcf().savefig(mysavedata + 'm_top.png')
        plt.gcf().clear()
'''

m = PlotHisto()
m.Plot_m_b_jf()
m.Plot_eta_jf()
m.Plot_q_lW()
m.Plot_eta_lW()
m.Plot_pT_W()
m.Plot_pT_lW()
m.Plot_m_Z()
m.Plot_eta_Z()
m.Plot_delR_jf_Z()
m.Plot_pT_jf()
m.Plot_pT_jr()
m.Plot_eta_jr()
m.Plot_pT_Z()
#m.Plot_mT_W()
#m.Plot_m_top()





