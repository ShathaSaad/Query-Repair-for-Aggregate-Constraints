import pandas as pd
import numpy as np
from numpy import arange,power
import matplotlib
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('PDF')
#import statsmodels.formula.api as sm




def tpch():
	name='ACSIncome_noOfCombination'
	dfpq1=pd.read_csv('/Users/Shatha/Downloads/Figs/ACSIncome_numOfCombination.csv',sep=",")
    
	group_label=list(dfpq1['datasize'].unique())
	pl.ioff()

	axpq1=dfpq1[['Q7','Q8','Q9','Q10']].plot.bar(width=0.6)

	legend = axpq1.legend(bbox_to_anchor=(-0.027, 1.036),prop={'size': 20},labels=['Q7','Q8','Q9','Q10'],loc=2,
              borderpad=0.1,labelspacing=0,handlelength=1,handletextpad=0.2,
              columnspacing=0.5,framealpha=1, ncol=2)
	legend.get_frame().set_edgecolor('black')
    
    # axis labels and tics
	axpq1.set_ylabel('Number of Combinations', fontsize=28)
	axpq1.set_xlabel('Dataset Size', fontsize=25) 

	axpq1.set_xticklabels(dfpq1['datasize'])
	pl.xticks(rotation=0)

	axpq1.tick_params(axis='x', labelsize=25)
	axpq1.tick_params(axis='y', labelsize=25)

	axpq1.set_yscale("log", nonpositive='clip')
	pl.ylim([0.1, 30000])

	# grid
	axpq1.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
	axpq1.set_axisbelow(True)
    # Add title
    

    # pl.show()
	pl.savefig(str(name) + '.pdf', bbox_inches='tight')
	pl.cla()


def main():
    tpch()   

if __name__=="__main__":
    main()
