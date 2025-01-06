from calendar import c
import time
from tkinter.tix import LabelEntry
import os
from distribution_calculation import distribution_calculation
from investigation_graphs import investigation_graphs
from SQL_operators import SQL_operators
from kd_tree1 import kd_tree1
from brute_force import brute_force
from Dataframe import Dataframe
from attributesRanges1 import attributesRanges1
from attributesRanges import attributesRanges
import pandas as pd
from genGraph import genGraph
import numpy as np
from ACSIncome_update import ACSIncome_update
from generate_clusters1 import generate_clusters1
from TreeNode import TreeNode
from generate_convex_hull import generate_convex_hull
from statistical_calculation import statistical_calculation
from filtered_fully import filtered_fully
from filtered_with_Ranges_generalize_topK1 import filtered_with_Ranges_generalize_topK1
from filtered_with_Ranges_generlize_nlogn import filtered_with_Ranges_generlize_nlogn
from filtered_partially import filtered_partially
from predicatesPossibleValues import predicatesPossibleValues
from ExpressionEvaluator import ExpressionEvaluator
from ExpressionEvaluator1 import ExpressionEvaluator1
from SyntheticDataGenerator import SyntheticDataGenerator
from parent_child_map import parent_child_map
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, uniform, skew, kurtosis
#from fitter import Fitter, get_common_distributions, get_distributions
#from distfit import distfit


def analyze_distribution(df, column):
    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(df[column], bins=30, edgecolor='black')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
 
def get_clusters(df_merged):
    start_time = time.time() 
    KD_tree = kd_tree1(df_merged, dim=4, bucket_size=15, branches=5)
    KD_tree_dict = KD_tree.flatten_tree()
    KD_tree.save_to_csv("KD_tree.csv")
    # Save the tree to a CSV file
    #generator1 = generate_clusters1()
    #cluster_tree1 = generator1.generating_clusters(df_merged.values)#, n_clusters=None, distance_threshold=20)  # Adjust as needed    
    #cleaned_clusters = generator1.remove_duplicates(cluster_tree1)
    #final_cluster_tree = generator1.add_metadata(cleaned_clusters, df_merged.values, cluster_tree1)
    end_time = time.time() 
    print("Generating clusters time: ", round(end_time - start_time, 3))

    return KD_tree_dict

def get_convex_hull(cluster_tree):
    convex_hull = generate_convex_hull()
    # Calculate and print convex hull points for each cluster in the tree
    hulls = convex_hull.calculate_convex_hulls_for_tree(cluster_tree)
    
    #for hull in hulls:
        #print(f"Convex Hull Points: {hull['Convex Hull Points']}, Level {hull['Level']}, Cluster {hull['Cluster Id']}")
    # Prepare hull info for CSV
    hull_info_list = []
    for hull in hulls:
        hull_info_list.append({
            'Level': hull['Level'],
            'Cluster Id': hull['Cluster Id'],
            'Data points': hull['Data points']
            #'Constraint points': hull['Constraint points']
        })

    # Create a DataFrame
    df_hulls = pd.DataFrame(hull_info_list)

    # Save to CSV
    df_hulls.to_csv('hull_info.csv', index=False)

    #convex_hull.draw_convex_hulls(cluster_tree)
    return hull_info_list

def get_statistical_info(cluster_tree, df, aggregations, predicates_number, constraint_columns, dataName, dataSize, query_num):
    stat_info = statistical_calculation()
    stat_tree = stat_info.statistical_calculation(cluster_tree, df, aggregations, predicates_number, constraint_columns, dataName, dataSize, query_num)

    return stat_tree

def userQuery_german():
    #getting dataframe
    input = Dataframe()
    df_original = input.getDataframe_german()

    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original,"credit_history", "==", "A33",  "A34")
    df2 = op.filter("UQ", df1, "investment_as_income_percentage", ">=", 2)
    df3 = op.filter("UQ", df2, "age", ">=", 26)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df3, all_pred_possible_values, op

def userQuery_ACSIncome_Q1(distribution, correlation, size, binSize):
    query_num = 1
    #getting dataframe
    input = Dataframe()
    correlated_pairs = []
    '''
    generator = SyntheticDataGenerator(seed=123)
    # Example: Generate dataset with custom attribute names and integer data
    attribute_specs = {
        'WKHP': (distribution, 1,100),       
        'SCHL': (distribution, 1, 25),                
        'COW': (distribution, 1, 9), 
        'PINCP': (distribution, 110, 1290000), 
        'SEX': (distribution, 1, 3)             
    }
    correlated_pairs = [ #[-0.3, 0.3, -0.25, 0.3]
    ('WKHP', 'SEX', correlation[0]),
    ('WKHP', 'PINCP', correlation[1]),
    ('SEX', 'PINCP', correlation[2])
    ]
    df_original, dataName, dataSize = generator.generate_synthetic_data2(
        n_samples=size, 
        attribute_specs=attribute_specs, 
        correlated_pairs=correlated_pairs, 
        bins=binSize  # Control unique values
    )
    '''
    df_original, dataName, dataSize = input.getDataframe_ACSIncome(size)
    print("\n\nData Size: ", dataSize)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 40) 
    df2 = op.filter("UQ", df1, "SCHL", ">=", 19) 
    df3 = op.filter("UQ", df2, "COW", "==", 4) 

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 
    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs

def userQuery_ACSIncome_Q2(distribution, correlation, size, binSize):
    query_num = 2
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome(size)

    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "COW", "==", 2)
    df2 = op.filter("UQ", df1, "PINCP", ">=", 60000.0)
    #df3 = op.filter("UQ", df2, "COW", ">=", 1)
    
    #data = pd.read_csv("/Users/Shatha/Downloads/Query_Refinment_Shatha/ACSIncome_state_number1_updated1.csv")
    #print(len(data["WKHP"].unique()), len(data["SCHL"].unique()), len(data["PINCP"].unique()))
    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs
    
def userQuery_ACSIncome_Q3(distribution, correlation, size, binSize):
    query_num = 3
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 60)
    df2 = op.filter("UQ", df1, "SCHL", ">=", 22)
    #df3 = op.filter("UQ", df2, "PINCP", ">", 60000.0)
    df3 = op.filter("UQ", df2, "COW", ">=", 2.0)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs  

def userQuery_ACSIncome_Q4(distribution, correlation, size, binSize):
    query_num = 4
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 60)
    df2 = op.filter("UQ", df1, "SCHL", "<=", 16)
    df3 = op.filter("UQ", df2, "PINCP", "<", 20000.0)
    
    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

def constraint1_ACSIncome(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("SEX == 1 and PINCP >= 20000")',
        "agg2":'count("SEX == 1")',
        "agg3": 'count("SEX == 2 and PINCP >= 20000")',
        "agg4": 'count("SEX == 2")'
    }
    expression = f"{constraint[0]} <= (agg1 / agg2) - (agg3 / agg4) <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def constraint2_ACSIncome(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("RAC1P == 1 and PINCP >= 20000")',
        "agg2":'count("RAC1P == 1")',
        "agg3": 'count("RAC1P == 2 and PINCP >= 20000")',
        "agg4": 'count("RAC1P == 2")'
    }
    expression = f"{constraint[0]} <= (agg1 / agg2) - (agg3 / agg4) <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def userQuery_Healthcare_Q1(distribution, correlation, size, binSize):
    query_num = 1
    #getting dataframe
    input = Dataframe()
    correlated_pairs = []
    '''
    generator = SyntheticDataGenerator(seed=123)
    # Example: Generate dataset with custom attribute names and integer data
    attribute_specs = {
        'income': (distribution, 1, 501),     
        'num-children': (distribution, 0, 6),        
        'county': (distribution, 0, 4), 
        'race': (distribution, 1, 5),                
        'label': (distribution, 110, 1290000)            
    }
    correlated_pairs = [
        ('income', 'label', correlation[0]),
        ('income', 'race', correlation[1]),
        ('race', 'label', correlation[2])
    ]
    
    df_original, dataName, dataSize = generator.generate_synthetic_data2(
        n_samples=size, 
        attribute_specs=attribute_specs, 
        correlated_pairs=correlated_pairs, 
        bins=binSize  # Control unique values
    )
    '''
    df_original, dataName, dataSize = input.getDataframe_Healthcare(size)
    print("\n\nData Size: ", dataSize)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 200.0) #it was not this value originally
    df2 = op.filter("UQ", df1, "num-children", ">=", 3)
    df3 = op.filter("UQ", df2, "county", "<=", 3)#, 'county4')

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs   

def userQuery_Healthcare_Q2(distribution, correlation, size, bin):
    #-0.3 <= SPD <= 0.3 #(race1, race2) +label
    query_num = 2
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 150)
    df2 = op.filter("UQ", df1, "complications", "<=", 8)
    df3 = op.filter("UQ", df2, "num-children", "<=", 4)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs 

def userQuery_Healthcare_Q3(distribution, correlation, size, bin):
    query_num = 3
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 300000)
    df2 = op.filter("UQ", df1, "complications", ">=", 5)
    df3 = op.filter("UQ", df2, "county", "==", 1)#, "county4")  

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs      

def userQuery_Healthcare_Q4(distribution, correlation, size, bin):
    query_num = 4
    correlated_pairs = []
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 150000)
    df2 = op.filter("UQ", df1, "num-children", "<=", 4)
    df3 = op.filter("UQ", df2, "complications", "<=", 8)
    df4 = op.filter("UQ", df3, "county", "==", 2)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df4, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs   

def constraint1_Healthcare(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("race == 1 and label == 1")',
        "agg2":'count("race == 1")',
        "agg3": 'count("race == 2 and label == 1")',
        "agg4": 'count("race == 2")'
    }
    expression = f"{constraint[0]} <= (agg1 / agg2) - (agg3 / agg4) <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def constraint2_Healthcare(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("ageGroup == 1 and label == 1")',
        "agg2":'count("ageGroup == 1")',
        "agg3": 'count("ageGroup == 2 and label == 1")',
        "agg4": 'count("ageGroup == 2")'
    }
    expression = f"{constraint[0]} <= (agg1 / agg2) - (agg3 / agg4) <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def constraint2_Cardinality(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator1()
    aggregations = {
        "agg1": 'count("race == 2")',
        "agg2": 'count("race == 1")'
    }
    constraint_type = 2
    expression = f"agg1 - agg2 <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression, constraint_type

def userQuery_TPCH_Q1(distribution, correlation, size, bin):
    query_num = 1
    #getting dataframe
    input = Dataframe()
    correlated_pairs = []
    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    print(df_lineitem)
    
    # Perform the join operations
    merged_df = pd.merge(pd.merge(df_orders, df_customer, left_on='o_custkey', right_on='c_custkey'), df_lineitem, left_on='o_orderkey', right_on='l_orderkey')
    merged_df1 = op.filter("UQ", merged_df, "c_mktsegment", "==", "BUILDING")
    merged_df2 = op.filter("UQ", merged_df1, "o_orderdate", "<", '1995-03-28')
    merged_df3 = op.filter("UQ", merged_df2, "l_shipdate", ">", '1995-03-28')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs

def userQuery_TPCH_Q3():
    query_num = 1
    #getting dataframe
    input = Dataframe()
    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH()
    op = SQL_operators()
    predAttRanges = attributesRanges()

    # Perform the join operations
    merged_df = pd.merge(pd.merge(df_partsupp, df_supplier, left_on='ps_suppkey', right_on='s_suppkey'), df_nation, left_on='s_nationkey', right_on='n_nationkey')

    merged_df1 = op.filter("UQ", merged_df, "n_name", "<", "NATION")
    merged_df2 = op.filter("UQ", merged_df1, "l_shipdate", "<", "l_commitdate") #Not accept comparing values of 2 columns
    merged_df3 = op.filter("UQ", merged_df2, "l_shipmode", "==", "RAIL", "AIR")
    merged_df4 = op.filter("UQ", merged_df3, "l_receiptdate", "<", '1995-01-01')
    merged_df5 = op.filter("UQ", merged_df4, "l_shipdate", ">", '1996-01-01')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df5, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery_TPCH_Q2_1(distribution, correlation, size, bin):
    query_num = 1
    #getting dataframe
    input = Dataframe()
    correlated_pairs = []

    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH(size)
    op = SQL_operators()
    predAttRanges = attributesRanges()

    # Perform the join operations
    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(df_part, df_partsupp, left_on='p_partkey', right_on='ps_partkey'), df_supplier, left_on='ps_suppkey', right_on='s_suppkey'), 
    df_nation, left_on='s_nationkey', right_on='n_nationkey'), df_region, left_on='n_regionkey', right_on='r_regionkey')
    # Save the merged DataFrame to a CSV file
    #output_file = 'merged_data.csv'
    #merged_df.to_csv(output_file, index=False)

    merged_df1 = op.filter("UQ", merged_df, "p_size", ">=", 10)
    merged_df2 = op.filter("UQ", merged_df1, "p_type", ">=", 32) 
    merged_df3 = op.filter("UQ", merged_df2, "r_name", ">=", 4)

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs 

def userQuery_TPCH_Q2_2():
    query_num = 2
    #getting dataframe
    input = Dataframe()
    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH("1GB")
    op = SQL_operators()
    predAttRanges = attributesRanges()

    # Perform the join operations
    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(df_part, df_partsupp, left_on='p_partkey', right_on='ps_partkey'), df_supplier, left_on='ps_suppkey', right_on='s_suppkey'), 
    df_nation, left_on='s_nationkey', right_on='n_nationkey'), df_region, left_on='n_regionkey', right_on='r_regionkey')


    merged_df1 = op.filter("UQ", merged_df, "p_size", "<=", 30)
    merged_df2 = op.filter("UQ", merged_df1, "p_type", ">=", 30)
    merged_df3 = op.filter("UQ", merged_df2, "r_name", "<", 3)

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num

def constraint1_TPCH(df, constraint): 
    result_list = []
    evaluator = ExpressionEvaluator1()
    aggregations = {
        "agg1": 'sum("p_retailprice")',
        "agg2":'count("p_retailprice")',
        "agg3": 'sum("s_acctbal")',
        "agg4": 'count("s_acctbal")'
    }
    expression = f"{constraint[0]} <= (agg3 / agg4) -(agg1 / agg2) <= {constraint[1]}"

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def whyNotProperty1(filtered_df):
    op = SQL_operators()
    #male_positive = len(op.filter("WhyN", df, "sex", "== Male", "label", '== yes'))
    #female_positive = len(op.filter("WhyN", df, "sex", "== Female", "label", '== yes'))
    #male_count = len(op.filter("WhyN", df, "sex", "== Male")) #count with filter condition
    #female_count = len(op.filter("WhyN", df, "sex", "== Female")) #count with filter condition
    male_positive_tuples_sex = op.filter("WhyN", filtered_df, "SEX", "==", 1.0) 
    male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, "PINCP", ">=", 20000.0) 
    female_positive_tuples_sex = op.filter("WhyN", filtered_df, "SEX", "==", 2.0)
    female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "PINCP", ">=", 20000.0) 
    male_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 1.0) #count with filter condition
    female_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 2.0) #count with filter condition
    male_positive = len(male_positive_tuples)
    female_positive = len(female_positive_tuples)
    male_count = len(male_count_tuples)
    female_count = len(female_count_tuples) 

    try:
        SPD = (male_positive / male_count) - (female_positive / female_count) #arithmatic expression
        print("***** Original Query SPD: ", round(SPD, 5), "\n\n")
        #print("Male ratio: ", male_positive / male_count, " Female ratio: ", female_positive / female_count," The SPD= ", SPD)
    except ZeroDivisionError:
        SPD = "Nan" 
        print("Nan")
    return op.getUserWhyNot()

def whyNotProperty2(df):
    op = SQL_operators()
    count_CS = len(op.filter("WhyN", df, "Major", "== CS", "Major", "== EE"))
    df_CS = op.filter("WhyN", df, "Major", "== CS", "Major", "== EE")
    Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
    try:
        avg = Sum_GPA_CS / count_CS  #arithmatic expression
        print("Sum : ", Sum_GPA_CS, "Count : ", count_CS,  "AVERAGE = ", avg)
    except ZeroDivisionError:
        avg = "Nan" 
    return op.getUserWhyNot()

def whyNotProperty_german(df):
    op = SQL_operators()
    #male_positive = len(op.filter("WhyN", df, "sex", "== Male", "label", '== yes'))
    #female_positive = len(op.filter("WhyN", df, "sex", "== Female", "label", '== yes'))
    #male_count = len(op.filter("WhyN", df, "sex", "== Male")) #count with filter condition
    #female_count = len(op.filter("WhyN", df, "sex", "== Female")) #count with filter condition
    male_positive = len(op.filter("WhyN", df, "personal_status", "== male", "credit", "== yes"))
    female_positive = len(op.filter("WhyN", df, "personal_status", "== female", "credit", "== yes"))
    male_count = len(op.filter("WhyN", df, "personal_status", "== male")) #count with filter condition
    female_count = len(op.filter("WhyN", df, "personal_status", "== female")) #count with filter condition
    try:
        SPD = (male_positive / male_count) - (female_positive / female_count) #arithmatic expression
        print("Original data SPD: ", round(SPD, 2), "\n\n")
        #print("Male ratio: ", male_positive / male_count, " Female ratio: ", female_positive / female_count," The SPD= ", SPD)
    except ZeroDivisionError:
        SPD = "Nan" 
    return op.getUserWhyNot()

def main():
    
    #ACSIncome_update_att = ACSIncome_update()
    #ACSIncome_update_att.att_update("WKHP", 35)
    # Define the possible configurations
    distributions = ['Non']#, 'normal', 'exponential'] 
    correlations = [[0.0, 0.0]]#, [0.3, 0.2], [0.8, 0.7]]
    constraints = [[0, 600]]
    #[[0.25, 0.5], [0.44, 0.5], [0.42, 0.5], [0.35, 0.5], [0.5, 0.6], [0.14, 0.17], [0.14, 0.15], [0.064, 0.069], [0.48, 0.5], [0.49, 0.5], [0.31, 0.36], [0.45, 0.54], [0.36, 0.43]]

    k = 7
    sizes = [50000] #'1MB', '10MB', '100MB', '1GB'

    bin_sizes = [0]
    
    for distribution in distributions:
        for correlation in correlations:
            for size in sizes:
                for bin in bin_sizes:
                    df_original, column_names, all_pred_possible_values, sorted_possible_refinments1, predicate_att_list, op = [], [], [], [], [], []
                    df_original, df_userQueryOut, all_pred_possible_values, op, dataName, dataSize, query_num, correlated_pairs =  userQuery_Healthcare_Q1(distribution, correlation, size, bin)        
                    predicate_att_list = op.getPredicateList()
                    column_names = [item[0] for item in predicate_att_list]     # Extract the column names from the predicate_att_list     
                    PCL_list = brute_force() 
                    possibleValues = predicatesPossibleValues()
                    sorted_possible_refinments1 = possibleValues.generate_possible_refinments_similarity(all_pred_possible_values, op.getPredicateList())
                    combination = len(sorted_possible_refinments1)
                    for constraint in constraints:
                        #for k in top_k:
                            count = 1
                            while count <= 1:
                                constraint_columns, df_merged, df_constraint, corr_matrix, df_predicate, statistical_tree, cluster_tree, columns = [], [], [], [], [], [], [], []
                                df_predicate = pd.DataFrame(df_original, columns=column_names)      # Convert to a Pandas DataFrame with dynamic column names
                                constraint_columns, aggregations, expression, constraint_type = constraint2_Cardinality(df_original, constraint) # Extract the column names, aggregations and expression from the constraint
                                df_constraint = pd.DataFrame(df_original, columns=constraint_columns)    
                                df_merged = pd.concat([df_predicate, df_constraint], axis=1)        # Merging the dataframes (predicates columns with constraints columns) by their index
                                cluster_tree = get_clusters(df_merged.values.tolist())
                                stat_start_time = time.time() 
                                statistical_tree = get_statistical_info(cluster_tree, df_merged, aggregations, len(all_pred_possible_values), constraint_columns, dataName, dataSize, query_num)
                                stat_end_time = time.time()
                                print("Number of Combinations: ", combination)
                                print("Time of collecting Statistical information:", round(stat_end_time - stat_start_time, 4))

                                #for col in df_merged.columns:
                                    #print(f"\nAnalyzing distribution for {col}")
                                    #analyze_distribution(df_merged, col)

                                corr_matrix = df_merged.corr()
                                print('\n\nPredicates and Constraints correlation:\n-------------------------------------------------------\n', corr_matrix)

                                #whyNotProperty1(df_userQueryOut) #to calculate the SPD for the original data
                                
                                #hull_info_list = get_convex_hull(cluster_tree)
                                
                                print("\n\n-----------------------Brute Force------------------------")
                                #calling Possible Candidate Lists to go through all possible refinments
                                #PCL_list.PossibleRef_allCombination(df_merged, sorted_possible_refinments1, dataSize, dataName, k, len(all_pred_possible_values), constraint) 

                                print("\n\n--------------------------Full---------------------------\n")
                                filter_fully = filtered_fully()
                                filter_fully.check_predicates(sorted_possible_refinments1, statistical_tree, expression, dataSize, dataName, k, query_num, constraint, op.getPredicateList(), combination, distribution, correlated_pairs, constraint_type)

                                print("\n\n--------------------Partial with Ranges-------------------\n")
                                ranges = attributesRanges1()
                                filter_ranges_partial = filtered_with_Ranges_generalize_topK1()
                                filter_ranges_partial1 = filtered_with_Ranges_generlize_nlogn()
                                all_pred_possible_Ranges= ranges.generatePossibleValues_equalWidth1(df_original, op.getPredicateList(), sorted_possible_refinments1)  
                                filter_ranges_partial.check_predicates(statistical_tree, all_pred_possible_Ranges, sorted_possible_refinments1, expression, dataSize, dataName, k, op.getPredicateList(), query_num, constraint, combination, distribution, correlated_pairs, constraint_type)
                                
                                # Generate a unique ID based on configuration
                                #config_id = f"size{size}_constr{constraint[0]}-{constraint[1]}_dist{distribution}_corr{correlation[0]}{correlation[1]}"
                                config_id = f"size{size}_constr{constraint[0]}-{constraint[1]}"

                                # Log the results
                                print("----------------------------------------------------------------------------\n")
                                #print(f"Completed run with Running configuration Id: {config_id}, size={size}, constraints={constraint}, distribution={distribution}, correlation={correlation}, bin={bin}, top-K={k}\n")
                                print(f"Completed run with Running configuration Id: {config_id}, size={size}, constraints={constraint}, top-K={k}\n")
                                print("*********************************************************************************************\n")
                                #calling whyNotProperty method
                                #UserWhyNot = whyNotProperty_german(df_userQueryOut)
                                count +=1
    
    
    '''
    #graph.generate_all_graphs(name)
    #graph.GeneratGraph_Find_Result_Percent(name, directory_path)
    #graph.constraint_width(name, directory_path)
    #graph.Density(name, directory_path)
    #graph.Solution_Num(name, directory_path)
    #graph.range_partial_time(name, directory_path)
    #graph.time_difference(name, directory_path)
    
    # Loop through constraints and process files
    for constraint in constraints:
        # Format the constraint as a string for the file name
        constraint_str = f"[{constraint[0]}, {constraint[1]}]"
        file_name = f"satisfied_conditions_Fully_Healthcare_size20000_query1_constraint{constraint_str}"
        file_path = os.path.join(directory_path, file_name)
        graph.GeneratGraph_Clusters_3D(file_name)
        # Check if file exists
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            # Add your processing logic here
        else:
            print(f"File not found: {file_path}")
    
    
    #graph.GeneratGraph_Clusters_Similarity() 
    #graph.GeneratGraph_Time_taken(name)
    #graph.GeneratGraph_Clusters_access(name)
    #graph.GeneratGraph_Constraints_checked(name)

    
    
    # Load the data
    file_path = "/Users/Shatha/Downloads/TPC-H/tpch_0_01/region.csv"  # Replace with the actual file path
    df = pd.read_csv(file_path)

    # Create a mapping for 'p_type' values to consistent numbers
    unique_values = df['r_name'].unique()
    p_type_mapping = {value: idx + 1 for idx, value in enumerate(sorted(unique_values))}

    # Map the 'p_type' column to numerical values based on the mapping
    df['r_name_numeric'] = df['r_name'].map(p_type_mapping)

    # Save the mapping for reference if needed
    mapping_file = "r_name_mapping.csv"
    pd.DataFrame(list(p_type_mapping.items()), columns=["r_name", "numeric_value"]).to_csv(mapping_file, index=False)

    # Save the updated file
    output_file = "region.csv"  # Specify the output file path
    df.to_csv(output_file, index=False)

    print(f"Updated file saved to {output_file}")
    print(f"Mapping saved to {mapping_file}")
    

    
    distribution_cal = distribution_calculation()
    #distribution_cal.distribution_calculation('satisfied_conditions_Fully_Healthcare_size300000_query1_constraint[0.2, 0.3]')
    
    # File path to the dataset
    
    
    directory_path = '/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final2/Time_vs_Constraints_A'  # Update this to your directory path
    graph = genGraph(directory_path)
    
    Measures = ['Time', 'Checked Num', 'Access Num']
    inv_graphs = investigation_graphs()
    for m in Measures:
        inv_graphs.Generate_Time_vs_Constraints(m)
        #graph.Generate_Time_vs_Constraints(m)
        #inv_graphs.plot_time_by_constraint_Distance(m)
        #inv_graphs.plot_time_by_constraint_Num_Solutions(m)
        #inv_graphs.plot_time_by_constraint_Width(m)
    '''
if __name__ == '__main__':

    main()

