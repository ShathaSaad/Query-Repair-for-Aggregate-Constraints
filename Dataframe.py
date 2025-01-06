import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, expon, beta

class Dataframe:
    def __init__(self, seed=42):
        np.random.seed(seed)  # Set the random seed for reproducibility

    def getDataframe_german(self):
        #df_original = pd.read_csv('/Users/Shatha/Downloads/Adult-sample1.csv')
        df_original = pd.read_csv('/Users/Shatha/Downloads/ApplicantsData.csv')
        #df_original = pd.read_csv('/Users/Shatha/Library/CloudStorage/OneDrive-UniversityofSouthampton/My_Phd_Code/Data/german4.csv')
        #df_original = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/customer.cpy',  sep='|')
        dataName= "toyExample"
        dataSize= '$$$'
        
        return df_original, dataName, dataSize

    def getDataframe_TPCH(self, size):
        dataName= "TPC-H"
        
        if size == "1MB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/customer.csv')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/lineitem.csv')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/nation.csv')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/orders.csv')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/part.csv')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/partsupp.csv')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/region.csv')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/supplier.csv')
        elif size == "10MB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/customer.csv')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/lineitem.csv')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/nation.csv')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/orders.csv')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/part.csv')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/partsupp.csv')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/region.csv')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/supplier.csv')
        elif size == "100MB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/customer.csv')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/lineitem.csv')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/nation.csv')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/orders.csv')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/part.csv')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/partsupp.csv')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/region.csv')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/supplier.csv')
        elif size == "1GB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/customer.csv')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/lineitem.csv')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/nation.csv')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/orders.csv')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/part.csv')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/partsupp.csv')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/region.csv')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/supplier.csv')
        
        # Filter related tables using keys from the sampled primary table
        # Sample the primary table (e.g., df_part)
        '''
        df_part_sampled = df_part.head(size)  # Order-preserving sampling
        df_partsupp_sampled = df_partsupp[df_partsupp['ps_partkey'].isin(df_part_sampled['p_partkey'])]
        df_supplier_sampled = df_supplier[df_supplier['s_suppkey'].isin(df_partsupp_sampled['ps_suppkey'])]
        df_nation_sampled = df_nation[df_nation['n_nationkey'].isin(df_supplier_sampled['s_nationkey'])]
        df_region_sampled = df_region[df_region['r_regionkey'].isin(df_nation_sampled['n_regionkey'])]
        df_customer_sampled = df_customer[df_customer['c_nationkey'].isin(df_nation_sampled['n_nationkey'])]
        df_orders_sampled = df_orders[df_orders['o_custkey'].isin(df_customer_sampled['c_custkey'])]
        df_lineitem_sampled = df_lineitem[df_lineitem['l_orderkey'].isin(df_orders_sampled['o_orderkey'])]
        '''
        return df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, size


    def getDataframe_ACSIncome(self,size):
        dataName= "ACSIncome"
        df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/ACSIncome_state_number1.csv')
        #profile = ProfileReport(df_original, title="Pandas Profiling Report")
        #print(profile)
        df = df_original.sample(n= size, random_state=42)  # You can change the value of n as needed

        if size == "1M":
            dataSize= 1048576
            return df_original, dataName, dataSize
        elif size == "100K":
            # Sample 100k rows from the DataFrame
            df = df_original.sample(n= 20000, random_state=42)  # You can change the value of n as needed
            dataSize= 100
        elif size == "500K":
            # Sample 500k rows from the DataFrame
            df = df_original.sample(n=500000, random_state=42)  # You can change the value of n as needed
            dataSize= 500000

        return df, dataName, size

    def getDataframe_Healthcare(self, size):
        dataName= "Healthcare"
        df = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_800_numerical.csv')
        df = df.sample(n=size, replace=True, random_state=42)  # You can change the value of n as needed
        #output_path = '/Users/Shatha/Downloads/inputData/healthcare_800_numerical_sampled.csv'
        #df.to_csv(output_path, index=False)
        dataSize= size
        if size == 12:
            df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_12_numerical.csv')
            dataSize= 12
        elif size == 100:
            df = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_100_numerical.csv')
            dataSize= 100
        elif size == 200:
            df = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_200_numerical.csv')
            dataSize= 200
        elif size == 400:
            df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_400_numerical.csv')
            dataSize= 400
        elif size == 600:
            df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_600_numerical.csv')
            dataSize= 600
        elif size == 800:
            df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/healthcare_800_numerical.csv')
            df = df_original.sample(n=20000, replace=True, random_state=42)  # You can change the value of n as needed
            dataSize= 800

        return df, dataName, size


    def generate_synthetic_data1(self, n_samples, attribute_specs=None, correlated_pairs=None, bins=None):
        """
        Generates synthetic integer data with specified attributes, distributions, and correlations for specified pairs.
        
        Parameters:
            n_samples (int): Number of samples to generate.
            attribute_specs (dict): Dictionary where keys are attribute names and values are distributions 
                                    ('normal', 'uniform', 'exponential', 'beta').
            correlated_pairs (list): List of tuples specifying attribute pairs and their desired correlation. 
                                    Example: [('Income', 'Age', 0.8), ('Income', 'Spending_Score', 0.6)]
            bins (int): Number of unique values (bins) to generate for each attribute.

        Returns:
            pd.DataFrame: DataFrame with generated synthetic data as integers.
        """

        attribute_names = list(attribute_specs.keys())
        synthetic_data = {}

        # Step 1: Generate independent data for attributes not in correlated pairs
        independent_attributes = set(attribute_names)
        for pair in correlated_pairs:
            independent_attributes -= set(pair[:2])

        for attr in independent_attributes:
            dist_type, low, high = attribute_specs[attr]
            synthetic_data[attr] = self.generate_attribute(n_samples, dist_type, low, high, bins)

        # Step 2: Generate correlated pairs
        for attr1, attr2, corr in correlated_pairs:
            if attr1 in synthetic_data or attr2 in synthetic_data:
                raise ValueError(f"Attributes '{attr1}' and '{attr2}' must be unique in correlated pairs.")
            
            dist_type1, low1, high1 = attribute_specs[attr1]
            dist_type2, low2, high2 = attribute_specs[attr2]

            # Generate correlated normal data
            mean = [0, 0]
            cov = [[1, corr], [corr, 1]]
            base_data = np.random.multivariate_normal(mean, cov, n_samples)

            # Transform correlated normal data to specified distributions
            synthetic_data[attr1] = self.transform_to_distribution(base_data[:, 0], dist_type1, low1, high1, bins)
            synthetic_data[attr2] = self.transform_to_distribution(base_data[:, 1], dist_type2, low2, high2, bins)

        # Step 3: Convert to DataFrame and save
        df = pd.DataFrame(synthetic_data)
        file_path = 'synthetic_data_correlated.csv'
        df.to_csv(file_path, index=False)
        print(f"Synthetic integer data with specified correlations saved to {file_path}")
        return df, "synthetic_data", n_samples

    def generate_attribute(self, n_samples, dist_type, low, high, bins=None):
        """Generate an independent attribute based on a specified distribution."""
        if dist_type == 'normal':
            data = norm.rvs(size=n_samples)
        elif dist_type == 'uniform':
            data = uniform.rvs(size=n_samples)
        elif dist_type == 'exponential':
            data = expon.rvs(size=n_samples)
        elif dist_type == 'beta':
            data = beta.rvs(2, 5, size=n_samples)
        else:
            raise ValueError("Supported distributions: 'normal', 'uniform', 'exponential', 'beta'")
        
        scaled_data = data * (high - low) + low

        # Apply binning if specified
        if bins and isinstance(bins, int):
            bins_range = np.linspace(low, high, bins)
            binned_data = np.digitize(scaled_data, bins=bins_range, right=True)
            # Scale each bin to ensure it falls within the range [low, high]
            return np.clip(binned_data, low, high).astype(int)

        return np.round(scaled_data).astype(int)

    def transform_to_distribution(self, data, dist_type, low, high, bins=None):
        """Transform standard normal data to a specified distribution and scale to integer range with binning."""
        if dist_type == 'normal':
            transformed_data = norm.cdf(data) * (high - low) + low
        elif dist_type == 'uniform':
            transformed_data = uniform.cdf(data) * (high - low) + low
        elif dist_type == 'exponential':
            transformed_data = expon.ppf(norm.cdf(data)) * (high - low) + low
        elif dist_type == 'beta':
            transformed_data = beta.ppf(norm.cdf(data), a=2, b=5) * (high - low) + low
        else:
            raise ValueError("Supported distributions: 'normal', 'uniform', 'exponential', 'beta'")
        
        # Apply binning if specified
        if bins and isinstance(bins, int):
            bins_range = np.linspace(low, high, bins)
            binned_data = np.digitize(transformed_data, bins=bins_range, right=True)
            return np.clip(binned_data, low, high).astype(int)

        return np.clip(np.round(transformed_data), low, high).astype(int)