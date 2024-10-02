import pandas as pd

class Dataframe:

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
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/customer.cpy',  sep='|')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/lineitem.cpy',  sep='|')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/nation.cpy',  sep='|')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/orders.cpy',  sep='|')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/part.cpy',  sep='|')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/partsupp.cpy',  sep='|')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/region.cpy',  sep='|')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_001/supplier.cpy',  sep='|')
            dataSize= "1MB"
        elif size == "10MB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/customer.cpy',  sep='|')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/lineitem.cpy',  sep='|')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/nation.cpy',  sep='|')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/orders.cpy',  sep='|')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/part.cpy',  sep='|')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/partsupp.cpy',  sep='|')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/region.cpy',  sep='|')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_01/supplier.cpy',  sep='|')
            dataSize= "10MB"
        elif size == "100MB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/customer.cpy',  sep='|')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/lineitem.cpy',  sep='|')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/nation.cpy',  sep='|')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/orders.cpy',  sep='|')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/part.cpy',  sep='|')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/partsupp.cpy',  sep='|')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/region.cpy',  sep='|')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_0_1/supplier.cpy',  sep='|')
            dataSize= "100MB"
        elif size == "1GB":
            df_customer = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/customer.cpy',  sep='|')
            df_lineitem = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/lineitem.cpy',  sep='|')
            df_nation = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/nation.cpy',  sep='|')
            df_orders = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/orders.cpy',  sep='|')
            df_part = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/part.cpy',  sep='|')
            df_partsupp = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/partsupp.cpy',  sep='|')
            df_region = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/region.cpy',  sep='|')
            df_supplier = pd.read_csv('/Users/Shatha/Downloads/TPC-H/tpch_1/supplier.cpy',  sep='|')
            dataSize= "1GB"


        return df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize


    def getDataframe_ACSIncome(self,size):
        dataName= "ACSIncome"
        df_original = pd.read_csv('/Users/Shatha/Downloads/inputData/ACSIncome_state_number1.csv')
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

        return df, dataName, dataSize





