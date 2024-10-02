import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

class generate_clusters1:

    def generating_clusters(self, df_merged):
        # Perform hierarchical clustering
        Z = linkage(df_merged, method='complete', metric='euclidean')
        return self.get_clusters_at_each_level(Z, len(df_merged))

    # Function to get clusters and their data points at each hierarchical level
    def get_clusters_at_each_level(self, Z, num_data_points):
        all_clusters = {}
        for level in range(1, num_data_points):
            cluster_labels = fcluster(Z, level, criterion='maxclust')
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)

            all_clusters[level] = clusters      

        all_clusters[num_data_points] = {i+1: [i] for i in range(num_data_points)}
        return all_clusters

    def remove_duplicates(self, all_clusters):
        unique_clusters = {}
        cleaned_clusters = {}

        for level, clusters in all_clusters.items():
            cleaned_clusters[level] = {}
            for cluster_id, points in clusters.items():
                cluster_tuple = tuple(sorted(points))
                if cluster_tuple not in unique_clusters:
                    unique_clusters[cluster_tuple] = cluster_id
                    cleaned_clusters[level][cluster_id] = points

        return cleaned_clusters

    def add_metadata(self, cleaned_clusters, df_merged, all_clusters):
        cluster_tree = []

        for level, clusters in cleaned_clusters.items():
            for cluster_id, points in clusters.items():
                parent_level = None
                parent_id = None

                # Find the parent cluster in earlier levels
                for prev_level in range(level-1, 0, -1):
                    for prev_cluster_id, prev_points in cleaned_clusters[prev_level].items():
                        if any(point in prev_points for point in points):
                            parent_level = prev_level
                            parent_id = prev_cluster_id
                            break
                    if parent_id is not None:
                        break
                                # If no parent is found, set parent_level to 0
                if parent_level is None:
                    parent_level = 0
                points1 = all_clusters[level][cluster_id]
                cluster_data = [df_merged[i].tolist() for i in points1]  # Corrected this line
                # Add metadata
                cluster_info = {
                    'Level': level,
                    'Cluster Id': cluster_id,
                    'Parent level': parent_level,
                    'Parent cluster': parent_id,
                    'Data points': cluster_data
                }
                cluster_tree.append(cluster_info)

        # Convert the cluster tree to a DataFrame for easier handling and export
        df_cluster_tree = pd.DataFrame(cluster_tree)
        df_cluster_tree.to_csv('Final_cluster_tree_with_metadata.csv', index=False)

        return cluster_tree

# Example usage:
# df_merged = pd.read_csv('your_dataset.csv')  # Load your data
# generator = GenerateClusters1()
# all_clusters = generator.generating_clusters(df_merged)
# cleaned_clusters = generator.remove_duplicates(all_clusters)
# final_cluster_tree = generator.add_metadata(cleaned_clusters)







