from collections import defaultdict
from unittest import skip
import pandas as pd
import time
from constraint_evaluation1 import constraint_evaluation1
from operators import operators
from filtered_partially import filtered_partially
from constraint_evaluation import constraint_evaluation



class filtered_with_Ranges1:
    def __init__(self):
        # Initialization code here
        self.applyOperator = operators()
        self.evaluate_constraint1 = constraint_evaluation1()  
        self.evaluate_constraint = constraint_evaluation()
        self.filter_fully =  filtered_partially()

    def divide_range(self, condition1Range, condition2Range, concrete_values1, concrete_values2):
        new_ranges1 = []
        new_ranges2 = []
        
        cond1_min, cond1_max = condition1Range
        cond2_min, cond2_max = condition2Range
        
        # Divide condition1 range
        if cond1_max - cond1_min > 0:
            mid_point1 = (cond1_min + cond1_max) // 2
            range1_left = (cond1_min, mid_point1)
            range1_right = (mid_point1 + 1, cond1_max)

            # Filter concrete values for each new range
            concrete_in_range1_left = [val for val in concrete_values1 if range1_left[0] <= val <= range1_left[1]]
            concrete_in_range1_right = [val for val in concrete_values1 if range1_right[0] <= val <= range1_right[1]]

            # Append the new range if it contains concrete values
            if concrete_in_range1_left:
                new_ranges1.append({'range': range1_left, 'concrete_values': concrete_in_range1_left})
            if concrete_in_range1_right:
                new_ranges1.append({'range': range1_right, 'concrete_values': concrete_in_range1_right})
        else:
            # If the range cannot be split, store the entire range with its concrete values
            concrete_in_range1 = [val for val in concrete_values1 if cond1_min <= val <= cond1_max]
            if concrete_in_range1:
                new_ranges1.append({'range': (cond1_min, cond1_max), 'concrete_values': concrete_in_range1})

        # Divide condition2 range
        if cond2_max - cond2_min > 0:
            mid_point2 = (cond2_min + cond2_max) // 2
            range2_left = (cond2_min, mid_point2)
            range2_right = (mid_point2 + 1, cond2_max)

            # Filter concrete values for each new range
            concrete_in_range2_left = [val for val in concrete_values2 if range2_left[0] <= val <= range2_left[1]]
            concrete_in_range2_right = [val for val in concrete_values2 if range2_right[0] <= val <= range2_right[1]]

            # Append the new range if it contains concrete values
            if concrete_in_range2_left:
                new_ranges2.append({'range': range2_left, 'concrete_values': concrete_in_range2_left})
            if concrete_in_range2_right:
                new_ranges2.append({'range': range2_right, 'concrete_values': concrete_in_range2_right})
        else:
            # If the range cannot be split, store the entire range with its concrete values
            concrete_in_range2 = [val for val in concrete_values2 if cond2_min <= val <= cond2_max]
            if concrete_in_range2:
                new_ranges2.append({'range': (cond2_min, cond2_max), 'concrete_values': concrete_in_range2})

        return new_ranges1, new_ranges2


    def check_predicates(self, statistical_tree, all_pred_possible_Ranges, expression, sorted_possible_refinments1):
        satisfied_conditions = []
        Concrete_values = []
        agg_counter = 0
        counter = 0
        child_counter = 0
        ranges_counter = 0
        found = False

        # Create dictionaries for parent-child relationships and cluster info
        parent_child_map = defaultdict(list)
        cluster_map = {}

        for row in statistical_tree:
            parent_key = (row['Parent level'], row['Parent cluster'])
            child_key = (row['Level'], row['Cluster Id'])
            parent_child_map[parent_key].append(child_key)
            cluster_map[child_key] = row

        # Find the root clusters (clusters with Parent level 0)
        root_clusters = [(row['Level'], row['Cluster Id']) for row in statistical_tree if row['Parent level'] == 0]

        start_time = time.time() 
        
        # Extract the first and second predicates
        first_predicate = all_pred_possible_Ranges[0]
        second_predicate = all_pred_possible_Ranges[1]

        # Get operators
        operator1 = first_predicate['operator']
        operator2 = second_predicate['operator']
        
        conditions_list = (first_predicate['values'],second_predicate['values'])
        conditions_list = list(conditions_list)
        # Initialize the loop index
        index1 = 0
        index2 = 0

       # Loop while there are new ranges to process in conditions_list
        while index1 < len(conditions_list[0]):
            condition1 = conditions_list[0][index1]
            #concrete_values1 = first_predicate['Concrete Vlaues'][index1]
            #if found == True:
                #break
            index2 = 0
            while index2 < len(conditions_list[1]):
                condition2 = conditions_list[1][index2]
                #concrete_values2 = second_predicate['Concrete Vlaues'][index2]
                #if found == True:
                    #break
                ranges_counter += 1
                filtered_clusters = []
                Concrete_values = []

                for root_key in root_clusters:

                    filtered_clusters, counter, child_counter = self.filter_clusters_partial_modified(
                        root_key, parent_child_map, cluster_map, filtered_clusters,
                        condition1, operator1, condition2, operator2, counter, child_counter)


                # Display the filtered clusters
                filtered_clusters_list_df = pd.DataFrame(filtered_clusters)
                #satisfied, agg_counter = self.evaluate_constraint.cardinality(filtered_clusters_list_df, counter, agg_counter, condition1, condition2)
                satisfied, agg_counter = self.evaluate_constraint1.calculate_expression_partially(filtered_clusters_list_df, condition1, condition2, agg_counter, expression, "ranges",  "similarity")                 
                print("----------------------", condition1, condition2)
                print("---------------------------1", satisfied)
                if satisfied != [] and satisfied['Range Satisfaction'] == 'Full':
                        
                        '''
                        for val1, val2 in zip(most_similar1['Concrete values'], most_similar2['Concrete values']):
                            Concrete_values = {
                                "condition1": condition1, 
                                "condition2": condition2,
                                'condition1': val1,  
                                'condition2': val2,
                                'Result': satisfied['Result'],
                                'Range Satisfaction': 'Full'}
                        print(Concrete_values)
                        '''
                        satisfied_conditions.append(satisfied)
                        #if len(satisfied_conditions) == 7:
                            #found = True
                            #break
                    
                else:
                    if (condition1[1] - condition1[0]) > 0 or (condition2[1] - condition2[0]) > 0:   
                        new_condition1, new_condition2 = self.divide_range(condition1, condition2, [10,5], [4,6])
                        for cond1 in new_condition1:
                            if cond1['range'] not in conditions_list[0]:
                                conditions_list[0].append(cond1['range'])
                            for cond2 in new_condition2:
                                if cond2['range'] not in conditions_list[1]:
                                    conditions_list[1].append(cond2['range'])
                    else: 
                        filtered_clusters = []                                       
                        filtered_clusters, counter = self.filter_fully.filter_clusters_Hash(root_key, parent_child_map, cluster_map, condition1[0], operator1, condition2[0], operator2, counter)
                        #new_satisfied, agg_counter = self.evaluate_constraint.cardinality(filtered_clusters, counter, agg_counter, range1['range'][0], range2['range'][0],)
                        satisfied, agg_counter = self.evaluate_constraint.evaluate_constraint(filtered_clusters, expression, condition1[0], condition2[0], agg_counter, " ")
                        print("--------------------------4", satisfied)
                        if satisfied != []:     
                            satisfied_conditions.append(satisfied)
                                #if len(satisfied_conditions) == 7:
                                    #found = True
                                    #break
                index2 += 1
            index1 += 1
                
                #filename = f'filtered_Partial_Ranges_{condition1}_{condition2}.csv'
                #filename = filename.replace(" ", "_").replace(",", "_")
                # Save to CSV
                #filtered_clusters_list_df.to_csv(filename, index=False)

                  

                                        
        print(satisfied_conditions)
        end_time = time.time() 

        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions__Ranges_Partial.csv", index=False)

        print("Number of clusters access: ", counter)
        print("Number of child clusters access: ", child_counter)
        print("Number of Aggregation calculated: ", agg_counter)
        print("Number of Ranges: ", ranges_counter)
        elapsed_time = end_time - start_time
        print("Time taken Overall for Partial filtered clusters:", round(elapsed_time, 3), "seconds") 


    def filter_clusters_partial_modified(self, cluster_key, parent_child_map, cluster_map, filtered_clusters, condition1, operator1, condition2, operator2, counter, child_counter, coverage_threshold=75):
        stack = [cluster_key]
        lower_bound = 0
        upper_bound = cluster_map[cluster_key]['Count'] 

        cond1_min, cond1_max = condition1
        cond2_min, cond2_max = condition2

        while stack:
            counter += 1
            cluster_key = stack.pop()
            cluster_info = cluster_map[cluster_key]

            # Check if this cluster fully satisfies the conditions
            if self.applyOperator.apply_operator_ranges(cluster_info['Data_Min'][0], cluster_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Full") and self.applyOperator.apply_operator_ranges(
                cluster_info['Data_Min'][1], cluster_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Full"):
                cluster_info['Satisfy'] = 'Full' 
                filtered_clusters.append(cluster_info)
                lower_bound += cluster_info['Count']
                      
            # Partially satisfying
            elif self.applyOperator.apply_operator_ranges(cluster_info['Data_Min'][0],  cluster_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Partial") and self.applyOperator.apply_operator_ranges(
                cluster_info['Data_Min'][1], cluster_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Partial"): 
                added = False
                if cluster_key in parent_child_map: 
                    not_satysfying_count = 0
                    child_list = []
                    for child_key in parent_child_map[cluster_key]:
                        child_counter+=1
                        child_info = cluster_map[child_key]

                        # Fully satisfying child
                        if self.applyOperator.apply_operator_ranges(child_info['Data_Min'][0],  child_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Full") and self.applyOperator.apply_operator_ranges(
                            child_info['Data_Min'][1], child_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Full"):
                            lower_bound += child_info['Count']
                            child_info['Satisfy'] = 'Full'
                            child_list.append({"satysfying": "full", "child": child_info})

                        # Partial child
                        elif self.applyOperator.apply_operator_ranges(child_info['Data_Min'][0],  child_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Partial") and self.applyOperator.apply_operator_ranges(
                            child_info['Data_Min'][1], child_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Partial"): 
                            child_info['Satisfy'] = 'Partial'
                            child_list.append({"satysfying": "partial", "child": [child_key]})

                        # Not satisfying child
                        else:
                            upper_bound -= child_info['Count']
                            not_satysfying_count += child_info['Count']

                    satisfaction_percentage = (lower_bound / upper_bound) * 100 if upper_bound > 0 else 0
                    
                    if (satisfaction_percentage >= coverage_threshold and not_satysfying_count < cluster_info['Count']/2):
                        # The partial cluster satisfies the threshold
                        cluster_info['Satisfy'] = 'Partial'
                        filtered_clusters.append(cluster_info)
                    
                        
                    else:
                        for child in child_list:
                                if child['satysfying'] == "partial":
                                    stack.extend(child["child"])
                                    #filtered_clusters.append(child["child"])

                                elif child['satysfying'] == "full":
                                    filtered_clusters.append(child["child"])
                else:
                    cluster_info['Satisfy'] = 'Partial'
                    filtered_clusters.append(cluster_info)
                        
                    
            #Not satisfying cluster   
            else:
                upper_bound -= cluster_info['Count']

                # If the cluster does not satisfy the condition, we explore its children if any
                if cluster_key in parent_child_map:
                    stack.extend(parent_child_map[cluster_key])
        
        return filtered_clusters, counter, child_counter

