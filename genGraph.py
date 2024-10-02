import pandas as pd
import numpy as np
from numpy import arange,power
import matplotlib
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
matplotlib.use('PDF')
matplotlib.use('TkAgg') 

class genGraph:


    def generateGraph(self, data_name, query_num):
        queryNum = [1]
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        for num in queryNum:
            #name = f'{data_name.lower().replace(" ", "_")}_runtime_data_Q{num}'
            # Construct the full file path
            name='Fully_info'
            file_path_csv = os.path.join(directory_path, f'{name}.csv')
            # Read the CSV file
            dfpq1 = pd.read_csv(file_path_csv, sep=",")
            group_label=list(dfpq1['Data Name'])
            pl.ioff()

            labels = dfpq1['Data Size'].tolist() 
            axpq1=dfpq1.plot.bar(width=0.7)
            plt.xticks(range(len(labels)), labels)

            legend = axpq1.legend(bbox_to_anchor=(-0.027, 1.036),prop={'size': 20},labels=['20','10','15','20000'],loc=2,
                    borderpad=0.1,labelspacing=0,handlelength=1,handletextpad=0.2,
                    columnspacing=0.5,framealpha=1, ncol=2)
            legend.get_frame().set_edgecolor('black')
        
        # axis labels and tics
        axpq1.set_ylabel('Checked No.', fontsize=28)
        axpq1.set_xlabel('Data Size', fontsize=25) 
        #axpq1.set_xticklabels(dfpq1['DataSize'])
        
        for tick in axpq1.xaxis.get_major_ticks():
            tick.label.set_fontsize(28) 
        for tick in axpq1.yaxis.get_major_ticks():
            tick.label.set_fontsize(28) 
            
        pl.xticks(rotation=0)
        
        #axpq1.set_yscale("log", nonposy='clip')
	    # pl.ylim([0.01, max(dfpq1['100'] + dfpq1['1000'] + dfpq1['10000'])] )
        #pl.ylim([0.01, 3000])

	    # # second x-axis
	    # ax2 = axpq1.twiny()
	    # ax2.set_xlabel("Datasize",fontsize=25)
	    # ax2.xaxis.labelpad = 12
	    # ax2.set_xlim(0, 60)
	    # ax2.set_xticks([7, 18, 30, 42, 53])
	    # ax2.set_xticklabels(['1K','10K','100K','1M','20M'], fontsize=25)

	    # grid
        axpq1.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
        axpq1.set_axisbelow(True)

	    # second x-axis
	    #ax2 = axpq1.twiny()
	    #ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
	    #ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
	    #ax2.spines['bottom'].set_position(('outward', 80))

	    #ax2.set_xlabel("Dataset size",fontsize=28)
	    # ax2.xaxis.labelpad = 12
	    #ax2.set_xlim(0, 60)
	    #ax2.set_xticklabels(['29K','260K','2.6M','26M'],fontsize=28)
	    #ax2.set_xticks([7, 22, 37, 52])

	    #ax2.set_frame_on(True)
	    #ax2.patch.set_visible(False)
	    #ax2.spines["bottom"].set_visible(True)

        # pl.show()
        # Construct the full file path
        file_path_pdf = os.path.join(directory_path, f'{name}.pdf')
        pl.savefig(file_path_pdf, bbox_inches='tight')
        pl.cla()


    
    def GeneratGraph1(self, data_name, query_num):
        directory_path = '/Users/Shatha/Downloads/inputData'
        name = f'{data_name.lower().replace(" ", "_")}_runtime_data_Q{query_num}'
        # Construct the full file path
        file_path_csv = os.path.join(directory_path, f'{name}.csv')
        # Read the CSV file
        dfwnq2 = pd.read_csv(file_path_csv, sep=",")
        
        group_label=list(dfwnq2['Data Size'].unique())
	    # pl.ioff()ß

	    # axwnq2=dfwnq2[['100','1000','10000','FULL']].plot.bar(width=0.9)
        axwnq2=dfwnq2.plot.bar(width=0.9)
        
        legend = axwnq2.legend(bbox_to_anchor=(-0.027, 1.036),prop={'size': 15},labels=['Baseline', 'Incremintal Agg'],loc=2,
                borderpad=0.1,labelspacing=0,handlelength=1,handletextpad=0.2,
                columnspacing=0.5,framealpha=1,ncol=2)
                
        legend.get_frame().set_edgecolor('black')
    
        # axis labels and tics
        axwnq2.set_ylabel('Runtime (sec)', fontsize=15)
        axwnq2.set_xlabel('#Data Size', fontsize=15) 

	    # axwnq2.set_xscale("symlog",linthreshx=4e6)
	    # pl.xlim([min(dfwnq2.provSize),max(dfwnq2.provSize)])

        ##axwnq2.set_xticks(dfwnq2['Data Size'])
        #axwnq2.set_xticklabels(['100','200','400','600','800'],fontsize=22)   
	    #axwnq2.set_xticks([6, 16, 25, 35, 45, 55]) 
	    #axwnq2.set_xticklabels(dfwnq2.provSize)
        # Set y-axis ticks and labels
        #axwnq2.set_yticks(dfwnq2['Time Taken'])  # Set y-axis ticks based on 'Time Taken' column data
        axwnq2.set_yticklabels(dfwnq2['Time Taken'], fontsize=5)  # Set y-axis tick labels based on 'Time Taken' column data with specified font size


	    #labels = [item.get_text() for item in axwnq2.get_xticklabels()]
	    #labels[0] = '1e11'
	    #labels[1] = '1e12'
	    #labels[2] = '1e14'
	    #labels[3] = '6e14'
	    #labels[4] = '1e15'
        axwnq2.set_xticklabels(dfwnq2['Data Size'], fontsize = 5)

	    #axwnq2.tick_params(labelsize=10)
        pl.xticks(rotation=0)

        #axwnq2.set_yscale("log", nonposy='clip')
	    # pl.ylim([0.01, max(dfwnq2['100'] + dfwnq2['1000'] + dfwnq2['10000'])])
        #pl.ylim([0.01,3000])

        for tick in axwnq2.xaxis.get_major_ticks():
	        tick.label.set_fontsize(10) 
        for tick in axwnq2.yaxis.get_major_ticks():
            tick.label.set_fontsize(10) 

	    # second x-axis
	    # ax2 = axwnq2.twiny()
	    # ax2.set_xlabel("Datasize",fontsize=25)
	    # ax2.xaxis.labelpad = 12
	    # ax2.set_xlim(0, 60)
	    # ax2.set_xticks([7, 18, 30, 42, 53])
	    # ax2.set_xticklabels(['1K','10K','100K','1M','20M'], fontsize=25)

	    # grid
        axwnq2.yaxis.grid(which='major',linewidth=3.0,linestyle=':')
        axwnq2.set_axisbelow(True)

	    # second x-axis
        #ax2 = axwnq2.twiny()
        #ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        #ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        #ax2.spines['bottom'].set_position(('outward', 80))

        #ax2.set_xlabel("Dataset size",fontsize=28)
	    # ax2.xaxis.labelpad = 12
        #ax2.set_xlim(0, 60)
        #ax2.set_xticklabels(['100','200','400','600','800'],fontsize=28)
        #ax2.set_xticks([7, 22, 37, 52])

        #ax2.set_frame_on(True)
        #ax2.patch.set_visible(False)
        #ax2.spines["bottom"].set_visible(True)

        # pl.show()
        file_path_pdf = os.path.join(directory_path, f'{name}.pdf')
        pl.savefig(file_path_pdf, bbox_inches='tight')
        pl.cla()
    

