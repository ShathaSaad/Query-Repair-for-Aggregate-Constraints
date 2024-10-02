from calendar import c
import time

from SQL_operators import SQL_operators
from kd_tree1 import kd_tree1
from brute_force import brute_force
from Dataframe import Dataframe
from attributesRanges import attributesRanges
import pandas as pd
from genGraph import genGraph
import numpy as np
from generate_clusters1 import generate_clusters1
from TreeNode import TreeNode
from generate_convex_hull import generate_convex_hull
from statistical_calculation import statistical_calculation
from filtered_fully import filtered_fully
from filtered_with_Ranges1 import filtered_with_Ranges1
from filtered_with_Ranges import filtered_with_Ranges
from filtered_partially import filtered_partially
from predicatesPossibleValues import predicatesPossibleValues
from ExpressionEvaluator import ExpressionEvaluator
from parent_child_map import parent_child_map

def get_clusters(df_merged):
    start_time = time.time() 
    KD_tree = kd_tree1(df_merged, dim=2, bucket_size=15, branches=5)
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

def get_statistical_info(cluster_tree, df, aggregations, predicates_number, constraint_columns):
    stat_info = statistical_calculation()
    stat_tree = stat_info.statistical_calculation(cluster_tree, df, aggregations, predicates_number, constraint_columns)

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

def userQuery_ACSIncome_Q1(size):
    #-0.2 <= float(SPD) <= 0.5
    query_num = 1
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome(size)
    print("\n\nData Size: ", dataSize)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 70) #73
    df2 = op.filter("UQ", df1, "SCHL", ">=", 12) #24
    #df3 = op.filter("UQ", df2, "COW", "==", 3.0) #, 4.0, 5.0) #8

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery_ACSIncome_Q2():
    # -0.5 <= float(SPD) <= 0.5:
    query_num = 2
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome("100K")
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter(type="UQ", df1=df_original, column1="WKHP", operator1="<=", value1=40)
    df2 = op.filter(type="UQ", df1=df1, column1="SCHL", operator1="<=", value1=19)
    #df3 = op.filter(type="UQ", df1=df2, column1="PINCP", operator1="<=", value1=40000.0)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num
    

def userQuery_ACSIncome_Q3():
    query_num = 3
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome("100K")
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 60)
    df2 = op.filter("UQ", df1, "SCHL", "<=", 16)
    #df3 = op.filter("UQ", df2, "PINCP", "<", 20000.0)
    
    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery_ACSIncome_Q4():
    query_num = 4
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_ACSIncome("100K")
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "WKHP", ">=", 60)
    df2 = op.filter("UQ", df1, "SCHL", ">=", 22)
    #df3 = op.filter("UQ", df2, "PINCP", ">", 60000.0)
    #df4 = op.filter("UQ", df3, "COW", "==", 6.0, 7.0)
    # Check for NaN values

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num  

def constraint1_ACSIncome(df): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("PINCP >= 20000 and SEX == 1")',
        "agg2":'count("SEX == 1")',
        "agg3": 'count("PINCP >= 20000 and SEX == 2")',
        "agg4": 'count("SEX == 2")'
    }
    expression = '0.0 <= (agg1 / agg2) - (agg3 / agg4) <= 0.1'

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def userQuery_Healthcare_Q1():
    query_num = 1
    #getting dataframe
    input = Dataframe()

    df_original, dataName, dataSize = input.getDataframe_Healthcare(800)
    print("\n\nData Size: ", dataSize)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 100) #it was not this value originally
    df2 = op.filter("UQ", df1, "num-children", ">=", 2)
    #df3 = op.filter("UQ", df2, "county", "==", 2)#, 'county4')

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num   

def userQuery_Healthcare_Q2():
    #-0.3 <= SPD <= 0.3 #(race1, race2) +label
    query_num = 2
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(800)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", "<=", 1000)
    df2 = op.filter("UQ", df1, "complications", ">=", 5)
    #df3 = op.filter("UQ", df2, "num-children", ">=", 4)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df2, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery_Healthcare_Q3():
    #-0.2 <= SPD <= 0.5 #(race1, race2) +label
    #if -0.5 <= float(SPD) <= 0.5:
    query_num = 3
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(800)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 150000)
    df2 = op.filter("UQ", df1, "num-children", "<=", 4)
    df3 = op.filter("UQ", df2, "complications", "<=", 8)
    df4 = op.filter("UQ", df3, "county", "==", "county2", "county4")

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df4, all_pred_possible_values, op, dataName, dataSize, query_num  

def userQuery_Healthcare_Q4():
    #-1 <= SPD <= 1 #(race1, race2) +label
    query_num = 4
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_Healthcare(800)
    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original, "income", ">=", 300000)
    df2 = op.filter("UQ", df1, "complications", ">=", 5)
    df3 = op.filter("UQ", df2, "county", "==", "county1", "county4")  

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num     

def constraint1_Healthcare(df): 
    result_list = []
    evaluator = ExpressionEvaluator()
    aggregations = {
        "agg1": 'count("race == 1 and label == 1")',
        "agg2":'count("race == 1")',
        "agg3": 'count("race == 2 and label == 1")',
        "agg4": 'count("race == 2")'
    }
    expression = '0.0 <= (agg1 / agg2) - (agg3 / agg4) <= 0.2'

    # Evaluate individual aggregations
    for index, (agg_name, agg_func) in enumerate(aggregations.items(), start=1):
        result = evaluator.evaluate_aggregation(df, agg_func)
        result_list.append(f"agg{index}: {result}")
    
    # Retrieve list of all columns used in the queries
    columns_used = evaluator.get_columns_used()
    
    return columns_used, aggregations, expression

def userQuery_TPCH_Q1():
    query_num = 1
    #getting dataframe
    input = Dataframe()
    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH()
    op = SQL_operators()
    predAttRanges = attributesRanges()

    # Perform the join operations
    merged_df = pd.merge(pd.merge(df_orders, df_customer, left_on='o_custkey', right_on='c_custkey'), df_lineitem, left_on='o_orderkey', right_on='l_orderkey')

    merged_df1 = op.filter("UQ", merged_df, "c_mktsegment", "==", "BUILDING")
    merged_df2 = op.filter("UQ", merged_df1, "o_orderdate", "<", '1995-03-28')
    merged_df3 = op.filter("UQ", merged_df2, "l_shipdate", ">", '1995-03-28')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num

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
    #merged_df2 = op.filter("UQ", merged_df1, "l_shipdate", "<", "l_commitdate") #Not accept comparing values of 2 columns
    #merged_df3 = op.filter("UQ", merged_df2, "l_shipmode", "==", "RAIL", "AIR")
    #merged_df4 = op.filter("UQ", merged_df3, "l_receiptdate", "<", '1995-01-01')
    #merged_df5 = op.filter("UQ", merged_df4, "l_shipdate", ">", '1996-01-01')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df1, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery_TPCH_Q2_1():
    query_num = 1
    #getting dataframe
    input = Dataframe()
    df_customer, df_lineitem, df_nation, df_orders, df_part, df_partsupp, df_region, df_supplier, dataName, dataSize = input.getDataframe_TPCH('1GB')
    op = SQL_operators()
    predAttRanges = attributesRanges()

    # Perform the join operations
    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(df_part, df_partsupp, left_on='p_partkey', right_on='ps_partkey'), df_supplier, left_on='ps_suppkey', right_on='s_suppkey'), 
    df_nation, left_on='s_nationkey', right_on='n_nationkey'), df_region, left_on='n_regionkey', right_on='r_regionkey')


    merged_df1 = op.filter("UQ", merged_df, "p_size", ">=", 10)
    merged_df2 = op.filter("UQ", merged_df1, "p_type", "==", 'LARGE BRUSHED')
    merged_df3 = op.filter("UQ", merged_df2, "r_name", "==", 'EUROPE')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num

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
    merged_df2 = op.filter("UQ", merged_df1, "p_type", "==", 'LARGE BRUSHED')
    merged_df3 = op.filter("UQ", merged_df2, "r_name", "==", 'EUROPE', 'ASIA')

    all_pred_possible_values = predAttRanges.generatePossibleValues(merged_df, op.get_predicates_attributes()) 

    return merged_df, merged_df3, all_pred_possible_values, op, dataName, dataSize, query_num

def userQuery1():
    query_num = 1
    #getting dataframe
    input = Dataframe()
    df_original, dataName, dataSize = input.getDataframe_german()

    op = SQL_operators()
    predAttRanges = attributesRanges()
    df1 = op.filter("UQ", df_original,"Major", "==", "EE", "CS")
    df2 = op.filter("UQ", df1, "GPA", ">=", 3.85)
    df3 = op.filter("UQ", df2, "Age", "<=", 45)

    all_pred_possible_values = predAttRanges.generatePossibleValues(df_original, op.get_predicates_attributes()) 

    return df_original, df3, all_pred_possible_values, op, dataName, dataSize, query_num  

def userQuery2():
    #getting dataframe
    input = Dataframe()
    df_original = input.getDataframe()
    op = SQL_operators()
    #print(df_original)
    count_CS = len(op.filter("WhyN", df_original, "Major", "== CS", "Major", "== EE"))
    df_CS = op.filter("WhyN", df_original, "Major", "== CS", "Major", "== EE")
    Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
    try:
        avg = Sum_GPA_CS / count_CS  #arithmatic expression
            #print("1-Sum : ", Sum_GPA_CS, "Count : ", count_CS,  "AVERAGE = ", avg)
    except ZeroDivisionError:
        avg = "Nan" 
    op = SQL_operators()


    df1 = op.filter("UQ", df_original, "Age", ">= 25") 
        #print(df1)

        #count_CS = len(df1)
        #Sum_GPA_CS = sum(df1["GPA"]) #count with filter condition

        #count_CS = len(op.filter("WhyN", df1, "Major", "== CS", "Major", "== EE"))
        #df_CS = op.filter("WhyN", df1, "Major", "== CS", "Major", "== EE")
        #Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
        #print("Sum : ", Sum_GPA_CS, "Count : ", count_CS)
        
    count_CS = len(op.filter("WhyN", df1, "Major", "== CS", "Major", "== EE"))
    df_CS = op.filter("WhyN", df1, "Major", "== CS", "Major", "== EE")
    Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
    try:
        avg = Sum_GPA_CS / count_CS  #arithmatic expression
        print("1-Sum : ", Sum_GPA_CS, "Count : ", count_CS,  "AVERAGE = ", avg)
    except ZeroDivisionError:
        avg = "Nan" 

    df2 = op.filter("UQ", df1, "GPA", ">= 3.90")
    #print(df1)

    count_CS = len(op.filter("WhyN", df2, "Major", "== CS", "Major", "== EE"))
    df_CS = op.filter("WhyN", df2, "Major", "== CS", "Major", "== EE")
    Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
    try:
        avg = Sum_GPA_CS / count_CS  #arithmatic expression
        print("2-Sum : ", Sum_GPA_CS, "Count : ", count_CS,  "AVERAGE = ", avg)
    except ZeroDivisionError:
        avg = "Nan" 


    #count_CS = len(df2)
    #Sum_GPA_CS = sum(df2["GPA"]) #count with filter condition
    #print("Sum : ", Sum_GPA_CS, "Count : ", count_CS)

    df3 = op.filter("UQ", df2,"Major", "== EE","Major", "== ME")
    #print(df1)

    count_CS = len(op.filter("WhyN", df3, "Major", "== CS", "Major", "== EE"))
    df_CS = op.filter("WhyN", df3, "Major", "== CS", "Major", "== EE")
    Sum_GPA_CS = sum(df_CS["Age"]) #count with filter condition
    try:
        avg = Sum_GPA_CS / count_CS  #arithmatic expression
        print("3-Sum : ", Sum_GPA_CS, "Count : ", count_CS,  "AVERAGE = ", avg)
    except ZeroDivisionError:
        print("Nan" )

    #count_CS = len(df3)
    #Sum_GPA_CS = sum(df3["GPA"]) #count with filter condition
    #print("Sum : ", Sum_GPA_CS, "Count : ", count_CS)

    return df3, op.getUserQuery()

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
    #size = [10, 15, 20, 25]
    #for s in size:
    df_original, df_userQueryOut, all_pred_possible_values, op, dataName, dataSize, query_num =  userQuery_ACSIncome_Q1(100000)

    predicate_att_list = op.getPredicateList()
    #whyNotProperty1(df_userQueryOut) #to calculate the SPD for the original data

    column_names = [item[0] for item in predicate_att_list]     # Extract the column names from the predicate_att_list
    
    df_predicate = pd.DataFrame(df_original, columns=column_names)      # Convert to a Pandas DataFrame with dynamic column names

    constraint_columns, aggregations, expression = constraint1_ACSIncome(df_original) # Extract the column names, aggregations and expression from the constraint
    
    df_constraint = pd.DataFrame(df_original, columns=constraint_columns) 

    df_merged = pd.concat([df_predicate, df_constraint], axis=1)        # Merging the dataframes (predicates columns with constraints columns) by their index

    possibleValues = predicatesPossibleValues()
    sorted_possible_refinments = possibleValues.generate_possible_refinments_inorder(all_pred_possible_values, op.getPredicateList())
    sorted_possible_refinments1 = possibleValues.generate_possible_refinments_similarity(all_pred_possible_values, op.getPredicateList())

    cluster_tree = get_clusters(df_merged.values.tolist())
    #hull_info_list = get_convex_hull(cluster_tree)

    stat_start_time = time.time()
    statistical_tree = get_statistical_info(cluster_tree, df_merged, aggregations, len(all_pred_possible_values), constraint_columns)
    stat_end_time = time.time()
    print("Time of collecting Statistical information:", round(stat_end_time - stat_start_time, 4))
    parent_descendants = parent_child_map()

    descendants= parent_descendants.precompute_all_descendants(statistical_tree)
        
    print("\n\n-----------------------Brute Force------------------------")
    #calling Possible Candidate Lists to go through all possible refinments
    PCL_list = brute_force() 
    PCL_list.PossibleRef_allCombination(df_merged, all_pred_possible_values, expression, aggregations, op.getPredicateList(), datasize, dataName) 

    print("\n\n--------------------------Full---------------------------\n")
    filter_fully = filtered_fully()
    filter_fully.check_predicates(sorted_possible_refinments1, statistical_tree, expression, dataSize, dataName)

    print("\n\n--------------------Full Incremental---------------------\n")
    #filter_fully.check_predicates_inc(sorted_possible_refinments, statistical_tree, descendants, expression)   

    print("\n\n-----------------------Partial------------------------\n")
    filter_partially = filtered_partially()
    filter_partially.check_predicates_partial_modified(sorted_possible_refinments1, statistical_tree, expression, datasize, dataName)

    print("\n\n--------------------Partial Incremental----------------------\n")
    #filter_partially.check_predicates_partial_inc(sorted_possible_refinments, statistical_tree, descendants, expression)

    print("\n\n--------------------Partial with Ranges-------------------\n")
    ranges = attributesRanges()

    filter_ranges_partial = filtered_with_Ranges()
    all_pred_possible_Ranges= ranges.generatePossibleValues_equalWidth(df_original, op.getPredicateList(), sorted_possible_refinments1)
    filter_ranges_partial.check_predicates(statistical_tree, all_pred_possible_Ranges, expression, sorted_possible_refinments1, datasize, dataName)
    
    #calling whyNotProperty method
    #UserWhyNot = whyNotProperty_german(df_userQueryOut)

    graph = genGraph()
    graph.generateGraph('healthcare', 1)

    
if __name__ == '__main__':

    main()

