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

class investigation_graphs:

    def plot_time_by_constraint_Distance(self, measure):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2/Distance_A'
        file_path = 'Run_info_ACSIncome_size50000_H'

        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{file_path}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Clean and filter necessary columns
        df = df[['Data Name', 'Type', 'Query Num', 'Distance', measure]].dropna()

        # Sort the DataFrame by Distance
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
        df = df.sort_values(by='Distance')

        # Determine global x-axis range
        global_min = df['Distance'].min()
        global_max = df['Distance'].max()

        # Define a color scheme for measures
        color_pairs = {
            "Time": ['blue', 'orange'],
            "Checked Num": ['green', 'red'],
            "Access Num": ['purple', 'brown']
        }
        colors = color_pairs.get(measure, ['gray', 'black'])

        # Create a unique figure with 3 subplots (1 row, 3 columns)
        unique_queries = sorted(df['Query Num'].unique())
        fig, axes = plt.subplots(1, len(unique_queries), figsize=(15, 5), sharey=True)

        if len(unique_queries) == 1:  # Ensure axes are iterable for a single query
            axes = [axes]

        for i, query_num in enumerate(unique_queries):
            ax = axes[i]

            # Filter data for the current Query Num
            query_df = df[df['Query Num'] == query_num]

            # Group by Type and aggregate
            for j, (type_name, group) in enumerate(query_df.groupby('Type')):
                aggregated = group.groupby('Distance', as_index=False)[measure].mean()
                ax.plot(
                    aggregated['Distance'],
                    aggregated[measure],
                    marker='o',
                    linestyle='-',
                    color=colors[j % len(colors)],  # Alternate between the two colors
                    label=type_name
                )

            # Customize each subplot
            ax.set_title(f'Query {query_num}', fontsize=14)
            ax.set_xlabel('Distance', fontsize=12)
            if i == 0:
                ax.set_ylabel(measure, fontsize=12)
            ax.grid(True)
            ax.legend()

            # Ensure consistent x-axis scaling
            ax.set_xlim(global_min, global_max)

        # Add a global title and adjust layout
        fig.suptitle(f'{measure} vs Distance for Different Queries', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure as a PDF in the specified directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)  # Create the directory if it doesn't exist
        output_file = os.path.join(directory_path, f'{measure}_vs_constraint_distance_combined.pdf')
        plt.savefig(output_file)
        plt.show()

        print(f"Graph saved to: {output_file}")

    def plot_time_by_constraint_Num_Solutions(self, measure):
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2/Solutions_Count_A'
        file_path = 'Run_info_ACSIncome_size50000_H'
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{file_path}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Clean and filter necessary columns
        df = df[['Data Name', 'Type', 'Query Num', 'Solutions Count', measure]].dropna()

        # Sort the DataFrame by Solutions Count
        df['Solutions Count'] = pd.to_numeric(df['Solutions Count'], errors='coerce')
        df = df.sort_values(by='Solutions Count')

        # Determine global x-axis range
        global_min = df['Solutions Count'].min()
        global_max = df['Solutions Count'].max()

        # Define a color scheme for measures
        color_pairs = {
            "Time": ['blue', 'orange'],
            "Checked Num": ['green', 'red'],
            "Access Num": ['purple', 'brown']
        }
        colors = color_pairs.get(measure, ['gray', 'black'])

        # Create a unique figure with 3 subplots (1 row, 3 columns)
        unique_queries = sorted(df['Query Num'].unique())
        fig, axes = plt.subplots(1, len(unique_queries), figsize=(15, 5), sharey=True)

        if len(unique_queries) == 1:  # Ensure axes are iterable for a single query
            axes = [axes]

        for i, query_num in enumerate(unique_queries):
            ax = axes[i]

            # Filter data for the current Query Num
            query_df = df[df['Query Num'] == query_num]

            # Group by Type and aggregate
            for j, (type_name, group) in enumerate(query_df.groupby('Type')):
                aggregated = group.groupby('Solutions Count', as_index=False)[measure].mean()
                ax.plot(
                    aggregated['Solutions Count'],
                    aggregated[measure],
                    marker='o',
                    linestyle='-',
                    color=colors[j % len(colors)],  # Alternate between the two colors
                    label=type_name
                )

            # Customize each subplot
            ax.set_title(f'Query {query_num}', fontsize=14)
            ax.set_xlabel('Solutions Count', fontsize=12)
            if i == 0:
                ax.set_ylabel(measure, fontsize=12)
            ax.grid(True)
            ax.legend()

            # Ensure consistent x-axis scaling
            ax.set_xlim(global_min, global_max)

        # Add a global title and adjust layout
        fig.suptitle(f'{measure} vs Solutions Count for Different Queries', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure as a PDF in the specified directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)  # Create the directory if it doesn't exist
        output_file = os.path.join(directory_path, f'{measure}_vs_constraint_Solutions_Count_combined.pdf')
        plt.savefig(output_file)
        plt.show()

        print(f"Graph saved to: {output_file}")

    def plot_time_by_constraint_Width(self, measure):
        directory_path = "/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2"
        file_path = 'Run_info_ACSIncome_size50000_H'
        # Drop duplicate constraints to ensure each appears only once
        file_path_csv = os.path.join(directory_path, f'{file_path}.csv')

        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Clean and filter necessary columns
        df = df[['Data Name', 'Type', 'Query Num', 'Constraint Width', measure]].dropna()

        # Sort the DataFrame by Constraint Width
        df['Constraint Width'] = pd.to_numeric(df['Constraint Width'], errors='coerce')
        df = df.sort_values(by='Constraint Width')

        # Determine global x-axis range
        global_min = df['Constraint Width'].min()
        global_max = df['Constraint Width'].max()

        # Define a color scheme for measures
        color_pairs = {
            "Time": ['blue', 'orange'],
            "Checked Num": ['green', 'red'],
            "Access Num": ['purple', 'brown']
        }
        colors = color_pairs.get(measure, ['gray', 'black'])

        # Create a unique figure with subplots for each Query Num
        unique_queries = sorted(df['Query Num'].unique())
        fig, axes = plt.subplots(1, len(unique_queries), figsize=(15, 5), sharey=True)

        if len(unique_queries) == 1:  # Ensure axes are iterable for a single query
            axes = [axes]

        for i, query_num in enumerate(unique_queries):
            ax = axes[i]

            # Filter data for the current Query Num
            query_df = df[df['Query Num'] == query_num]

            # Group by Type and aggregate
            for j, (type_name, group) in enumerate(query_df.groupby('Type')):
                aggregated = group.groupby('Constraint Width', as_index=False)[measure].mean()
                ax.plot(
                    aggregated['Constraint Width'],
                    aggregated[measure],
                    marker='o',
                    linestyle='-',
                    color=colors[j % len(colors)],  # Alternate between the two colors
                    label=type_name
                )

            # Customize each subplot
            ax.set_title(f'Query {query_num}', fontsize=14)
            ax.set_xlabel('Constraint Width', fontsize=12)
            if i == 0:
                ax.set_ylabel(measure, fontsize=12)
            ax.grid(True)
            ax.legend()

            # Ensure consistent x-axis scaling
            ax.set_xlim(global_min, global_max)

        # Add a global title and adjust layout
        fig.suptitle(f'{measure} vs Constraint Width for Different Queries', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure as a PDF in the specified directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)  # Create the directory if it doesn't exist
        output_file = os.path.join(directory_path, f'{measure}_vs_constraint_Width_combined.pdf')
        plt.savefig(output_file)
        plt.show()

        print(f"Graph saved to: {output_file}")

    def Generate_Time_vs_Constraints(self, measure):
        # Load the CSV file into a DataFrame
        file_path = "/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2/Time_vs_Constraints_A"  # Update this path
        file_path_csv = os.path.join(file_path, 'Run_info_Healthcare_size50000_H_brute.csv')
        # Read the CSV file
        df = pd.read_csv(file_path_csv, sep=",")

        # Filter and clean the necessary columns
        df = df[['Query Num', 'Con Name', 'Type', measure]].dropna()

        # Filter the DataFrame for the specified Query Num
        df = df[df['Query Num'] == 1]

        # Pivot the data to prepare for plotting
        pivot_df = df.pivot_table(index=['Con Name'], columns='Type', values=measure, aggfunc='mean').reset_index()

        # Define color mapping for the measure
        color_mapping = {
            "Time": {"BruteForce": "grey", "Fully": "cyan", "Ranges": "magenta"},
            "Checked Num": {"BruteForce": "green", "Fully": "purple", "Ranges": "orange"},
            "Access Num": {"BruteForce": "red", "Fully": "blue", "Ranges": "brown"},
        }
        colors = color_mapping.get(measure, {"BruteForce": "pink", "Fully": "grey", "Ranges": "black"})

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 3))

        # Bar positions and labels
        x = range(len(pivot_df))
        labels = pivot_df['Con Name']

        # Plot bars for each type with assigned colors
        if 'BruteForce' in pivot_df:
            ax.bar(x, pivot_df['BruteForce'], width=0.3, label='BruteForce', align='center', color=colors["BruteForce"])
        if 'Fully' in pivot_df:
            ax.bar([pos + 0.3 for pos in x], pivot_df['Fully'], width=0.3, label='Fully', align='center', color=colors["Fully"])
        if 'Ranges' in pivot_df:
            ax.bar([pos + 0.6 for pos in x], pivot_df['Ranges'], width=0.3, label='Ranges', align='center', color=colors["Ranges"])
        

        # Customize the plot
        ax.set_title(f'Query: {1}', fontsize=22)
        ax.set_xlabel('Constraint', fontsize=22)
        ax.set_xticks([pos + 0.3 for pos in x])
        ax.set_xticklabels(labels, ha='right', fontsize=22)
        if measure == "Time":
            ax.set_ylabel('Runtime (sec)', fontsize=22)
        elif measure == "Checked Num":
            ax.set_ylabel('Number of \nConstraints Checked', fontsize=20)
        elif measure == "Access Num":
            ax.set_ylabel('Number of Boxes Accessed \nin Cluster Tree', fontsize=22)
        ax.legend(fontsize=18)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Add a global title and adjust layout
        #fig.suptitle('Query and Constraint (Healthcare Dataset)', fontsize=22)
        plt.tight_layout()

        # Save the figure as a PDF
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        output_file = os.path.join(file_path, f'comparison_fully_vs_ranges_bruteforce_query{1}_{measure}.pdf')
        plt.savefig(output_file)
        plt.show()

        print(f"Graph saved to: {output_file}")
