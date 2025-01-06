
import dataclasses
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import ast  # To safely evaluate the string representation of a list
import os


class distribution_calculation:

    def distribution_calculation(self, data):
        points = self.cal_point_density(data)
        self.cal_spread(points)
        self.nearest_neighbor_dist(points)
        #self.clustring(points)

    def cal_point_density2(self, data):
        # Load CSV data
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/Healthcare/all results'
        file_path_csv = os.path.join(directory_path, f'{data}.csv')

        # Read the CSV file
        data = pd.read_csv(file_path_csv, sep=",")

        # Extract coordinates
        data['conditions'] = data['conditions'].apply(ast.literal_eval)  # Convert string representation to a list
        points = np.array(data['conditions'].tolist())  # Convert to NumPy array

        # Check the range along each dimension
        min_values = np.min(points, axis=0)
        max_values = np.max(points, axis=0)

        print(f"Min values: {min_values}")
        print(f"Max values: {max_values}")

        # Length of the line segment in the first dimension (since other dimensions are constant)
        line_length = max_values[0] - min_values[0]
        print(f"Line Length: {line_length}")

        # Calculate point density
        num_points = len(points)
        density = num_points / line_length
        print(f"Number of Points: {num_points}")
        print(f"Point Density: {density}")

        return density

    def cal_point_density(self, data):
        # Load CSV data
        directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/Healthcare/all results'
        file_path_csv = os.path.join(directory_path, f'{data}.csv')

        # Read the CSV file
        data = pd.read_csv(file_path_csv, sep=",")

        # Extract coordinates
        data['conditions'] = data['conditions'].apply(ast.literal_eval)  # Convert string representation to a list

        # Split the points into separate columns
        data[['Condition1', 'Condition2', 'Condition3']] = pd.DataFrame(data['conditions'].tolist(), index=data.index)
        points = np.array(data['conditions'].tolist())  # Convert to NumPy array

        # Extract the points as a NumPy array or DataFrame
        #points = data[['Condition1', 'Condition2', 'Condition3']].values  # As a NumPy array
        #print(points)
        # OR
        #points_df = data[['Condition1', 'Condition2', 'Condition3']]  # As a DataFrame


        # Compute the convex hull volume
        #hull = ConvexHull(points)
        #volume = hull.volume


        # Calculate density
        #density1 = volume /len(points) 
        density2 = len(points) / ((max(points[0]) - min(points[0])) * (max(points[1]) - min(points[1]))  * (max(points[2]) - min(points[2])))
        #print("Point Density1:", round(density1, 5))
        print("Point Density2:", round(density2, 5))
        return points
    
    def cal_spread(self, points): #(Variance or Standard Deviation)
        # Check if points is a valid 2D array
        if points is None or len(points) == 0:
            raise ValueError("Points array is empty or None. Ensure it is properly initialized.")
        
        points = np.array(points)  # Ensure points is a NumPy array

        if points.ndim != 2:
            raise ValueError("Points array must be a 2D array (rows = points, columns = dimensions).")

        # Calculate variance and standard deviation along each axis
        variance = np.var(points, axis=0)
        std_dev = np.std(points, axis=0)

        print("Variance:", variance)
        print("Standard Deviation:", std_dev)

    
    def nearest_neighbor_dist(self, points):
        # Compute pairwise distances
        dist_matrix = distance_matrix(points, points)

        # Get the nearest neighbor distance for each point
        nearest_distances = np.sort(dist_matrix, axis=1)[:, 1]  # Exclude self-distance
        avg_nearest_neighbor_distance = np.mean(nearest_distances)

        print("Average Nearest Neighbor Distance:", avg_nearest_neighbor_distance)
    
    def clustring(self, points):
        # Use KMeans to find clusters
        kmeans = KMeans(n_clusters=3)  # Change number of clusters as needed
        kmeans.fit(points)
        labels = kmeans.labels_

        print("Cluster Centers:", kmeans.cluster_centers_)
        print("Cluster Labels:", labels)



    



