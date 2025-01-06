import pandas as pd
import numpy as np
from numpy import arange,power
import matplotlib
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import os
matplotlib.use('PDF')
matplotlib.use('TkAgg') 

class genGraph:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def generateGraph(self, name, directory_path):
        queryNum = [1]
        #directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        for num in queryNum:
            #name = f'{data_name.lower().replace(" ", "_")}_runtime_data_Q{num}'
            # Construct the full file path
            
            file_path_csv = os.path.join(directory_path, f'{name}.csv')
            # Read the CSV file
            dfpq1 = pd.read_csv(file_path_csv, sep=",")
            group_label=list(dfpq1['Data Name'])
            pl.ioff()

            labels = dfpq1['Data Size'].tolist() 
            axpq1=dfpq1.plot.bar(width=0.7)
            plt.xticks(range(len(labels)), labels)

            legend = axpq1.legend(bbox_to_anchor=(-0.027, 1.036),prop={'size': 20},labels=['10000','20000','50000'],loc=2,
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


    def read_and_prepare_data(self, name):
        # Read the CSV file
        file_path_csv = os.path.join(self.directory_path, f'{name}.csv')
        df = pd.read_csv(file_path_csv, sep=",")
        
        # Sort data by 'Constraint' for consistent grouping
        df.sort_values(by=['Constraint'], inplace=True)
        return df

    def generate_graph(self, df, metric, ylabel, title, name):
        # Extract unique constraints
        constraints = df['Constraint'].unique()
        
        # Create a layout: 3 plots in the first row, 4 in the second
        fig, axes = plt.subplots(9, 9, figsize=(10, 7), sharey=False)
        axes = axes.flatten()
        
        for i, constraint in enumerate(constraints):
            ax = axes[i]
            print(ax)
            subset = df[df['Constraint'] == constraint]
            
            # Data for "Fully" and "Ranges"
            fully_data = subset[subset['Type'] == 'Fully'][metric].values
            ranges_data = subset[subset['Type'] == 'Ranges'][metric].values
            
            x = np.array([0])  # Single x-position for this metric
            
            # Plot bars
            ax.bar(x - 0.2, fully_data, width=0.4, label='Fully' if i == 0 else "", alpha=0.7)
            ax.bar(x + 0.2, ranges_data, width=0.4, label='Ranges' if i == 0 else "", 
                edgecolor='black', fill=False, hatch='//')
            
            # Titles and labels
            ax.set_title(f"Constraint {constraint}", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([metric], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=12)
        
        # Remove unused subplots
        if len(constraints) < len(axes):
            for j in range(len(constraints), len(axes)):
                fig.delaxes(axes[j])
        
        # Add legend to the first subplot
        #axes[0].legend(loc='upper right', fontsize=12)
        
        # Add a common legend outside the plot
        fig.legend(
            ['Fully', 'Ranges'],
            loc='upper center',
            fontsize=12,
            bbox_to_anchor=(0.87, 0.97),  # Position below the graphs
            ncol=2  # Place legends horizontally
        )
        
        # Adjust layout and save the plot
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_path_pdf = os.path.join(self.directory_path, f'{name}_{metric.lower().replace(" ", "_")}_by_constraint.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')
        plt.show()

    def generate_all_graphs(self, name):
        df = self.read_and_prepare_data(name)
        
        # Generate Time plot
        self.generate_graph(df, 'Time', 'Time (s)', 'Time by Constraint', name)
        
        # Generate Checked No. plot
        self.generate_graph(df, 'Checked No.', 'Checked No.', 'Checked No. by Constraint', name)
        
        # Generate Access No. plot
        self.generate_graph(df, 'Access No.', 'Access No.', 'Access No. by Constraint', name)


    # Updated graph generation function to group by Data Name and Data Size
    def GeneratGraph_Time_taken(self, name):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        dfwnq2 = pd.read_csv(file_path_csv, sep=",")

        # Set the desired order of 'Data Size'
        data_size_order = [300000]#[10000, 20000, 50000]
        
        # Convert Data Size to ordered categories if needed
        dfwnq2['Data Size'] = pd.Categorical(dfwnq2['Data Size'], categories=data_size_order, ordered=True)
        
        # Sort the data by 'Data Name' and 'Data Size' for correct grouping
        dfwnq2.sort_values(by=['Data Name', 'Data Size'], inplace=True)
        
        # Pivot the DataFrame to get each type's Refinement No. per Data Size within each Data Name
        pivot_df = dfwnq2.pivot_table(index=['Constraint'], columns=['type'], values='Time') #columns=['type', 'Distribution'

        # Plot the data
        axwnq2 = pivot_df.plot(kind='bar', width=1, figsize=(20, 10), colormap='viridis')

        # Customize legend
        legend = axwnq2.legend(bbox_to_anchor=(.8, .8), loc='upper left', fontsize=14)
        legend.get_frame().set_edgecolor('black')

        # Set axis labels
        axwnq2.set_ylabel('Time', fontsize=15)
        axwnq2.set_xlabel('Data Name and Size', fontsize=15)

        # Set title
        #plt.title(f'Time Comparison for {data_name}', fontsize=16)
        plt.title(f'Comparison of Time taken', fontsize=16)

        # Rotate x-axis labels
        plt.xticks(rotation=20, ha='right')

        # Grid
        axwnq2.yaxis.grid(True)
        axwnq2.set_axisbelow(True)

        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'{name}_Time_taken_Graph.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        
        plt.show()

    # Updated graph generation function to group by Data Name and Data Size
    def GeneratGraph_Find_Result_Percent(self, name, directory_path):
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        dfwnq2 = pd.read_csv(file_path_csv, sep=",")

        # Set the desired order of 'Data Size'
        data_size_order = [30000]
        
        # Convert Data Size to ordered categories if needed
        dfwnq2['Data Size'] = pd.Categorical(dfwnq2['Data Size'], categories=data_size_order, ordered=True)
        
        # Sort the data by 'Data Name' and 'Data Size' for correct grouping
        dfwnq2.sort_values(by=['Data Name', 'Data Size'], inplace=True)
        
        # Pivot the DataFrame to get each type's Refinement No. per Data Size within each Data Name
        pivot_df = dfwnq2.pivot_table(index=['Constraint'], columns=['type'], values='Find Result Percent.')

        # Plot the data
        axwnq2 = pivot_df.plot(kind='bar', width=0.8, figsize=(14, 8), colormap='viridis')

        # Customize legend
        legend = axwnq2.legend(
            loc='upper center',
            bbox_to_anchor=(0.87, 1.1),  # Position the legend above the graph
            fontsize=12,
            ncol=2  # Place the legend in two columns for a compact layout
        )
        legend.get_frame().set_edgecolor('black')

        # Set axis labels
        axwnq2.set_ylabel('Find Result Percent.', fontsize=15)
        axwnq2.set_xlabel('Data Name and Size', fontsize=15)

        # Set title
        #plt.title(f'Time Comparison for {data_name}', fontsize=16)
        plt.title(f'Comparison of Finding Result Percent', fontsize=16)

        # Rotate x-axis labels
        plt.xticks(rotation=20, ha='right')

        # Grid
        axwnq2.yaxis.grid(True)
        axwnq2.set_axisbelow(True)

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to fit the legend above the graph


        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'{name}_Find_Result_Percent_Graph.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        
        plt.show()


    # Updated graph generation function to group by Data Name and Data Size
    def GeneratGraph_Constraints_checked(self, name):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        dfwnq2 = pd.read_csv(file_path_csv, sep=",")

        # Set the desired order of 'Data Size'
        data_size_order = [30000]
        
        # Convert Data Size to ordered categories if needed
        dfwnq2['Data Size'] = pd.Categorical(dfwnq2['Data Size'], categories=data_size_order, ordered=True)
        
        # Sort the data by 'Data Name' and 'Data Size' for correct grouping
        dfwnq2.sort_values(by=['Data Name', 'Data Size'], inplace=True)
        
        # Pivot the DataFrame to get each type's Refinement No. per Data Size within each Data Name
        pivot_df = dfwnq2.pivot_table(index=['Combinations No.', 'Constraint'], columns=['type'], values='Checked No.')

        # Plot the data
        axwnq2 = pivot_df.plot(kind='bar', width=1, figsize=(20, 10), colormap='viridis')

        # Customize legend
        legend = axwnq2.legend(bbox_to_anchor=(.8, .8), loc='upper left', fontsize=14)
        legend.get_frame().set_edgecolor('black')

        # Set axis labels
        axwnq2.set_ylabel('Checked No.', fontsize=15)
        axwnq2.set_xlabel('Data Name and Size', fontsize=15)

        # Set title
        #plt.title(f'Time Comparison for {data_name}', fontsize=16)
        plt.title(f'Comparison of Number of Constraints Checked', fontsize=16)
        # Rotate x-axis labels
        plt.xticks(rotation=20, ha='right')

        # Grid
        axwnq2.yaxis.grid(True)
        axwnq2.set_axisbelow(True)

        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'{name}_Constraints_checked_Graph.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        
        plt.show()


    # Updated graph generation function to group by Data Name and Data Size
    def GeneratGraph_Clusters_access(self, name):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        dfwnq2 = pd.read_csv(file_path_csv, sep=",")

        # Set the desired order of 'Data Size'
        data_size_order = [30000]
        
        # Convert Data Size to ordered categories if needed
        dfwnq2['Data Size'] = pd.Categorical(dfwnq2['Data Size'], categories=data_size_order, ordered=True)
        
        # Sort the data by 'Data Name' and 'Data Size' for correct grouping
        dfwnq2.sort_values(by=['Data Name', 'Data Size'], inplace=True)
        
        # Pivot the DataFrame to get each type's Refinement No. per Data Size within each Data Name
        pivot_df = dfwnq2.pivot_table(index=['Combinations No.', 'Constraint'], columns=['type'], values='Access No.')

        # Plot the data
        axwnq2 = pivot_df.plot(kind='bar', width=1, figsize=(20, 10), colormap='viridis')

        # Customize legend
        legend = axwnq2.legend(bbox_to_anchor=(.8, .8), loc='upper left', fontsize=14)
        legend.get_frame().set_edgecolor('black')

        # Set axis labels
        axwnq2.set_ylabel('Access No.', fontsize=15)
        axwnq2.set_xlabel('Data Name and Size', fontsize=15)

        # Set title
        plt.title(f'Comparison of Number of Clusters Accesses', fontsize=16)

        # Rotate x-axis labels
        plt.xticks(rotation=20, ha='right')

        # Grid
        axwnq2.yaxis.grid(True)
        axwnq2.set_axisbelow(True)

        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'{name}_Clusters_accesses_Graph.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        
        plt.show()

    # Updated graph generation function to group by Data Name and Data Size
    def GeneratGraph_Clusters_Similarity(self, name):
        directory_base = '/Users/Shatha/Downloads/Query_Refinment_Shatha'
        subfolder = 'Similarity vs. solutions count'
        directory_path = os.path.join(directory_base, subfolder)        
        name = 'satisfied_conditions_Fully_Healthcare_5k'
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Group the data by Similarity and count occurrences
        similarity_counts = df['Similarity'].value_counts().sort_index()

        # Plot the data
        fig, ax = plt.subplots(figsize=(14, 8))
        similarity_counts.plot(kind='bar', ax=ax, width=0.8, color='skyblue')

        # Customize x-axis ticks with increments of 5
        max_similarity = int(similarity_counts.index.max())  # Convert max_similarity to an integer
        x_ticks = range(0, max_similarity + 5, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha='right')

        # Set axis labels
        ax.set_xlabel('Similarity', fontsize=15)
        ax.set_ylabel('Count of Conditions', fontsize=15)

        # Set title
        plt.title('Similarity vs. Conditions Count', fontsize=16)

        # Grid
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        # Save the figure
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/Similarity vs. solutions count'
        os.makedirs(directory_path, exist_ok=True)
        file_path_pdf = os.path.join(directory_path, f'{name}_Similarity_vs_Counts_Graph.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        plt.show()

    def constraint_width(self, name, directory_path):
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")
        df_cleaned = df.drop_duplicates(subset=['Constraint Width', 'Constraint'])

        # Sort the DataFrame by Constraint Width
        df_sorted = df_cleaned.sort_values(by='Constraint')

        # Plot the Line Graph
        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted['Constraint'], df_sorted['Constraint Width'], color='b', width=0.6)

        # Add labels and titles
        plt.title('Constraint Width', fontsize=18)
        plt.xlabel('Constraint', fontsize=18)
        plt.ylabel('Constraint Width', fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(False)

        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'constraint_width.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show the plot
        plt.show()    

    # Updated graph generation function to group by Data Name and Data Size
    def Density(self, name, directory_path):
       
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")
        df_cleaned = df.drop_duplicates(subset=['Density', 'Constraint'])

        # Sort the DataFrame by Constraint Width
        df_sorted = df_cleaned.sort_values(by='Constraint')

        # Plot the Line Graph
        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted['Constraint'], df_sorted['Density'], color='b', width=0.6)


        # Add labels and titles
        plt.title('Density', fontsize=18)
        plt.xlabel('Constraint', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(False)


        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'Density.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show the plot
        plt.show() 
    
        # Updated graph generation function to group by Data Name and Data Size
    def Solution_Num(self, name, directory_path):
       
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")
        df_cleaned = df.drop_duplicates(subset=['Solutions Count', 'Constraint'])

        # Sort the DataFrame by Constraint Width
        df_sorted = df_cleaned.sort_values(by='Constraint')

        # Plot the Line Graph
        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted['Constraint'], df_sorted['Solutions Count'], color='g', width=0.6)


        # Add labels and titles
        plt.title('Solutions Count', fontsize=18)
        plt.xlabel('Constraint', fontsize=18)
        plt.ylabel('Solutions Count', fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(False)


        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'Solutions_Num.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show the plot
        plt.show() 
    
    def range_partial_time(self, name, directory_path):
       
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")
        df_cleaned = df.drop_duplicates(subset=['Partial Time', 'Constraint'])

        # Sort the DataFrame by Constraint Width
        df_sorted = df_cleaned.sort_values(by='Constraint')

        # Plot the Line Graph
        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted['Constraint'], df_sorted['Partial Time'], color='y', width=0.6)


        # Add labels and titles
        plt.title('Partial Time', fontsize=18)
        plt.xlabel('Constraint', fontsize=18)
        plt.ylabel('Partial Time', fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(False)


        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'Range_Partial Time.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show the plot
        plt.show() 

    def time_difference(self, name, directory_path):
       
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{name}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")
        df_cleaned = df.drop_duplicates(subset=['Time difference', 'Constraint'])

        # Sort the DataFrame by Constraint Width
        df_sorted = df_cleaned.sort_values(by='Constraint')

        # Plot the Line Graph
        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted['Constraint'], df_sorted['Time difference'], color='r', width=0.6)


        # Add labels and titles
        plt.title('Time difference between Range and Full algorithm', fontsize=18)
        plt.xlabel('Constraint', fontsize=14)
        plt.ylabel('Time difference', fontsize=14)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(False)


        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'Time difference between Range and Full algorithm.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show the plot
        plt.show() 

    def GeneratGraph_Clusters_3D(self, name):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/all_results_files/Specific1'
        file_path_csv = os.path.join(directory_path, f'{name}.csv')
        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")


        # Extract and clean up the 'conditions' column
        # Remove any non-numeric characters except commas and split into three parts
        df['conditions'] = df['conditions'].str.extract(r'\[(.*?)\]').squeeze()  # Extracts content within brackets if any
        conditions_split = df['conditions'].str.split(",", expand=True)

        # Rename the columns and convert them to numeric values
        conditions_split.columns = ['condition1', 'condition2', 'condition3']
        df[['condition1', 'condition2', 'condition3']] = conditions_split.apply(pd.to_numeric, errors='coerce')

        dot_size = 5  # Adjust the size here

        # Filter out any rows where the conditions couldn't be converted to numbers
        df = df.dropna(subset=['condition1', 'condition2', 'condition3'])

        # Plotting
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(df['condition1'], df['condition2'], df['condition3'], c='b', marker='o', s=dot_size)

        # Set labels
        ax.set_xlabel('Condition 1', fontsize=14)
        ax.set_ylabel('Condition 2', fontsize=14)
        ax.set_zlabel('Condition 3', fontsize=14)
        plt.title('3D Scatter Plot of Conditions', fontsize=16)

        # Save the figure
        file_path_pdf = os.path.join(directory_path, f'{name}.pdf')
        plt.savefig(file_path_pdf, bbox_inches='tight')

        # Show plot
        plt.show()

    def Generate_Time_vs_Constraints(self, measure):
        # Load the CSV file into a DataFrame
        file_path = "/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2/Time_vs_Constraints_A"  # Update this path
        file_path_csv = os.path.join(file_path, 'Run_info_ACSIncome_size50000_H.csv')
        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Filter and clean the necessary columns
        df = df[['Query Num', 'Con Name', 'Type', measure]].dropna()

        # Pivot the data to prepare for plotting
        pivot_df = df.pivot_table(index=['Query Num', 'Con Name'], columns='Type', values=measure, aggfunc='mean').reset_index()

        # Sort the pivoted DataFrame by 'Con Name'
        pivot_df['Con Name Numeric'] = pivot_df['Con Name'].str.extract('C(\d+)', expand=False).astype(int)
        pivot_df = pivot_df.sort_values(by='Con Name Numeric').drop(columns=['Con Name Numeric'])

        # Define color mapping for the measure
        color_mapping = {
            "Time": {"Fully": "blue", "Ranges": "green"},
            "Checked Num": {"Fully": "red", "Ranges": "orange"},
            "Access Num": {"Fully": "purple", "Ranges": "cyan"},
        }
        colors = color_mapping.get(measure, {"Fully": "grey", "Ranges": "black"})

        # Create a bar plot for each Query Num
        unique_queries = pivot_df['Query Num'].unique()
        fig, axes = plt.subplots(1, len(unique_queries), figsize=(15, 5), sharey=True)

        # Sort unique `Query Num` to ensure graphs are ordered correctly
        unique_queries = sorted(pivot_df['Query Num'].unique())

        for i, query_num in enumerate(unique_queries):
            ax = axes[i]
            query_data = pivot_df[pivot_df['Query Num'] == query_num]

            # Bar positions and labels
            x = range(len(query_data))
            labels = query_data['Con Name']
            
            # Plot bars for "Fully" and "Ranges" with assigned colors
            ax.bar(x, query_data['Fully'], width=0.4, label='Fully', align='center', color=colors["Fully"])
            ax.bar([pos + 0.4 for pos in x], query_data['Ranges'], width=0.4, label='Ranges', align='center', color=colors["Ranges"])
            
            # Customize the plot
            ax.set_title(f'Query: {query_num}', fontsize=20)
            ax.set_xlabel('Constraint', fontsize=20)
            ax.set_xticks([pos + 0.2 for pos in x])
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
            if i == 0:
                if measure == "Time":
                    ax.set_ylabel('Runtime (sec)', fontsize=20)
                elif measure == "Checked Num":
                    ax.set_ylabel('Count of Constraint\nchecked', fontsize=20)
                elif measure == "Access Num":
                    ax.set_ylabel('Count of Boxes Accessed\nin Cluster Tree', fontsize=20)
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Add a global title and adjust layout
        #fig.suptitle('Query and Constraint (Healthcare Dataset)', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure as a PDF
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        output_file = os.path.join(file_path, f'comparison_fully_vs_ranges_{measure}_ACSIncome.pdf')
        plt.savefig(output_file)
        plt.show()

        print(f"Graph saved to: {output_file}")
