import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class ACSIncome_update: 



    def att_update(self, column_to_cluster, num_clusters):

        # Load the data
        file_path = "/Users/Shatha/Downloads/Query_Refinment_Shatha/ACSIncome_state_number1_updated1.csv"  # Update with the correct file path
        sample_size = 50000  # Define the sample size

        # Load the dataset and sample rows
        data = pd.read_csv(file_path)
        sampled_data = data.sample(n=sample_size, random_state=42)

        # Extract the column to be clustered
        values = sampled_data[column_to_cluster].dropna().values.reshape(-1, 1)  # Reshape for clustering

        # Perform K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        sampled_data['Cluster'] = kmeans.fit_predict(values)

        # Map values to their respective cluster centers
        cluster_centers = kmeans.cluster_centers_.flatten()
        sampled_data[column_to_cluster] = sampled_data['Cluster'].map(lambda cluster: cluster_centers[cluster])

        # Drop the temporary cluster column
        sampled_data = sampled_data.drop(columns=['Cluster'])

        # Replace the original column in the full dataset
        data[column_to_cluster] = data[column_to_cluster].map(
            lambda x: round(cluster_centers[kmeans.predict(np.array(x).reshape(-1, 1))[0]]) if not np.isnan(x) else x
)


        # Save the updated dataset
        output_file = "ACSIncome_state_number1_updated1.csv"  # Update with the desired output file path
        data.to_csv(output_file, index=False)

        print(f"Updated file saved as {output_file}")

        

    def process_large_csv(self):
        """
        Process a large CSV file by sampling and applying Quantile Binning.
        
        Args:
        - file_path (str): Path to the input CSV file.
        - attribute (str): The name of the column to apply quantile binning.
        - sample_size (int): Number of rows to sample.
        - num_bins (int): Number of bins for quantile binning.
        - output_file (str): Path to save the updated CSV.
        """
        file_path = '/Users/Shatha/Downloads/inputData/ACSIncome_state_number1.csv'
        attribute = 'PINCP'
        num_bins = 150
        output_file = "ACSIncome_state_number1_sampled_binned.csv"
        sample_size = 200000
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Sample the dataset
        df_sampled = df.sample(n=sample_size, random_state=42)
        
        # Apply Quantile Binning on the sampled data
        bin_edges = np.percentile(df_sampled[attribute], np.linspace(0, 100, num_bins + 1))
        bin_labels = np.arange(1, num_bins + 1)  # Assign bin labels from 1 to num_bins
        
        # Digitize and assign bin labels
        df_sampled[f"{attribute}_binned"] = np.digitize(df_sampled[attribute], bin_edges, right=True)
        
        # Replace each bin with its median
        bin_medians = {
            bin_label: df_sampled[df_sampled[f"{attribute}_binned"] == bin_label][attribute].median()
            for bin_label in bin_labels
        }
        df_sampled[f"{attribute}_binned"] = df_sampled[f"{attribute}_binned"].map(bin_medians)
        
        # Save the updated sampled dataset to a new CSV
        df_sampled.to_csv(output_file, index=False)
        print(f"Processed file saved as {output_file}")


