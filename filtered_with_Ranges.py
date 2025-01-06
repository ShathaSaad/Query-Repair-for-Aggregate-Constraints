from collections import defaultdict
from unittest import skip
import pandas as pd
import time
from constraint_evaluation1 import constraint_evaluation1
from operators import operators
from filtered_fully import filtered_fully
from constraint_evaluation import constraint_evaluation



class filtered_with_Ranges1:
    def __init__(self):
        # Initialization code here
        self.applyOperator = operators()
        self.evaluate_constraint1 = constraint_evaluation1()  
        self.evaluate_constraint = constraint_evaluation()
        self.filter_fully =  filtered_fully()

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
            if concrete_in_range1_left != []:
                new_ranges1.append({'range': range1_left, 'concrete_values': concrete_in_range1_left})
            if concrete_in_range1_right != []:
                new_ranges1.append({'range': range1_right, 'concrete_values': concrete_in_range1_right})
        else:
            # If the range cannot be split, store the entire range with its concrete values
            concrete_in_range1 = [val for val in concrete_values1 if cond1_min <= val <= cond1_max]
            if concrete_in_range1 != []:
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
            if concrete_in_range2_left != []:
                new_ranges2.append({'range': range2_left, 'concrete_values': concrete_in_range2_left})
            if concrete_in_range2_right != []:
                new_ranges2.append({'range': range2_right, 'concrete_values': concrete_in_range2_right})
        else:
            # If the range cannot be split, store the entire range with its concrete values
            concrete_in_range2 = [val for val in concrete_values2 if cond2_min <= val <= cond2_max]
            if concrete_in_range2 != []:
                new_ranges2.append({'range': (cond2_min, cond2_max), 'concrete_values': concrete_in_range2})
        
        return new_ranges1, new_ranges2


    def check_predicates(self, statistical_tree, all_pred_possible_Ranges, expression, sorted_possible_refinments1, datasize, dataName):
        satisfied_conditions = []
        Concrete_values = []
        agg_counter = 0
        counter = 0
        child_counter = 0
        ranges_counter = 0
        found = False
        check_counter = 0
        refinement_counter = 0

        # Create dictionaries for parent-child relationships and cluster info
        parent_child_map = defaultdict(list)
        cluster_map = {}
        Concrete_values = []

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

        # Loop over all ranges for each predicate
        for condition1, concrete_values1 in zip(first_predicate['values'], first_predicate['Concrete Vlaues']):
            if found == True:
                break
            if concrete_values1 == []:
                continue
            for condition2, concrete_values2 in zip(second_predicate['values'], second_predicate['Concrete Vlaues']):
                if concrete_values2 == []:
                    continue
                if found == True:
                    break
                ranges_counter += 1
                filtered_clusters = []

                for root_key in root_clusters:
                    filtered_clusters, counter, child_counter = self.filter_clusters_partial_modified(
                        root_key, parent_child_map, cluster_map, filtered_clusters,
                        condition1, operator1, condition2, operator2, counter, child_counter)

                # Display the filtered clusters
                filtered_clusters_list_df = pd.DataFrame(filtered_clusters)
                #satisfied, agg_counter = self.evaluate_constraint.cardinality(filtered_clusters_list_df, counter, agg_counter, condition1, condition2)
                check_counter +=1 
                satisfied, agg_counter = self.evaluate_constraint1.calculate_expression_partially1(filtered_clusters_list_df, condition1, condition2, agg_counter, 
                expression, "ranges",  "similarity", concrete_values1, concrete_values2)   

                
                #filename = f'filtered_Ranges_{condition1}_{condition2}.csv'
                #filename = filename.replace(" ", "_").replace(",", "_")
                # Save to CSV
                #filtered_clusters_list_df.to_csv(filename, index=False)  

                if satisfied != [] and satisfied['Range Satisfaction'] == 'Full':
                        refinement_counter += 1
                        satisfied_conditions.append(satisfied)
                        for concrete_value1 in satisfied['Concrete Vlaues1']:
                            for concrete_value2 in satisfied['Concrete Vlaues2']:
                                for refinment in sorted_possible_refinments1:
                                    if concrete_value1 == refinment['refinements'][0]['value']:
                                        if concrete_value2 == refinment['refinements'][1]['value']:
                                            Concrete_values.append({"condition1": concrete_value1, "condition2": concrete_value2, 
                                            "Result": satisfied['Result'], "Similarity": refinment['distance']})

                                            #if len(Concrete_values) == 40:
                                                #found = True
                                                #break
                    
                elif satisfied != [] and satisfied['Range Satisfaction'] == 'Partial':
                        new_satisfactions = [(condition1, condition2)]

                        while new_satisfactions:
                            #if found == True:
                                #break
                            current_condition1, current_condition2 = new_satisfactions.pop(0)

                            new_condition1, new_condition2 = self.divide_range(current_condition1, current_condition2, concrete_values1, concrete_values2)

                            for range1 in new_condition1:
                                if found == True:
                                    break
                                for range2 in new_condition2:
                                    if found == True:
                                        break
                                    ranges_counter+=1
                                    filtered_clusters = []
                                    if (range1['range'][0] != range1['range'][1]) or (range2['range'][0] != range2['range'][1]):
                                        for root_key in root_clusters:
                                            filtered_clusters, counter, child_counter = self.filter_clusters_partial_modified(
                                                root_key, parent_child_map, cluster_map, filtered_clusters, range1['range'], operator1, range2['range'], operator2, counter, child_counter)
                                                        
                                        filtered_clusters_list_df = pd.DataFrame(filtered_clusters)
                                        #new_satisfied, agg_counter = self.evaluate_constraint.cardinality(filtered_clusters_list_df, counter, agg_counter, range1['range'], range2['range'])
                                        check_counter +=1 
                                        new_satisfied, agg_counter = self.evaluate_constraint1.calculate_expression_partially1(filtered_clusters_list_df, range1['range'], range2['range'], agg_counter, expression,  "ranges", 
                                        " ", range1['concrete_values'], range2['concrete_values'])
                                            
                                        r1 = range1['range']
                                        r2 = range2['range']
                                        
                                        #filename = f'filtered_Ranges_{r1}_{r2}.csv'
                                        #filename = filename.replace(" ", "_").replace(",", "_")
                                        # Save to CSV
                                        #filtered_clusters_list_df.to_csv(filename, index=False) 

                                        if new_satisfied != [] and new_satisfied['Range Satisfaction'] == 'Full':
                                            refinement_counter += 1
                                            satisfied_conditions.append(new_satisfied)
                                            for concrete_value1 in new_satisfied['Concrete Vlaues1']:
                                                for concrete_value2 in new_satisfied['Concrete Vlaues2']:
                                                    for refinment in sorted_possible_refinments1:
                                                        if concrete_value1 == refinment['refinements'][0]['value']:
                                                            if concrete_value2 == refinment['refinements'][1]['value']:
                                                                Concrete_values.append({"condition1": concrete_value1, "condition2": concrete_value2, 
                                                                "Result": new_satisfied['Result'], "Similarity": refinment['distance']})

                                                                #if len(Concrete_values) == 40:
                                                                    #found = True
                                                                    #break

                                        elif new_satisfied != [] and new_satisfied['Range Satisfaction'] == 'Partial':
                                            new_satisfactions.append((range1['range'], range2['range']))

                                    else: #if the range is minimal, no need to divide and filter fully
                                        filtered_clusters = []
                                        for root_key in root_clusters:
                                            filtered_clusters, counter = self.filter_fully.filter_clusters_Hash2(root_key, cluster_map, range1['range'][0], operator1, range2['range'][0], operator2, 
                                            filtered_clusters, counter, parent_child_map)
                                        
                                        filtered_clusters_list_df = pd.DataFrame(filtered_clusters)

                                        check_counter +=1 
                                        satisfied, agg_counter = self.evaluate_constraint.evaluate_constraint2(filtered_clusters, expression, range1['range'][0], range2['range'][0], agg_counter, " ",
                                            "ranges", range1['concrete_values'], range2['concrete_values'])
                                        r1 = range1['range'][0]
                                        r2 = range2['range'][0]
                                        
                                        #filename = f'filtered_Ranges_{r1}_{r2}.csv'
                                        #filename = filename.replace(" ", "_").replace(",", "_")
                                        # Save to CSV
                                        #filtered_clusters_list_df.to_csv(filename, index=False) 
                                        
                                        if satisfied != []:
                                            refinement_counter += 1
                                            satisfied_conditions.append(satisfied)
                                            for concrete_value1 in satisfied['Concrete Vlaues1']:
                                                for concrete_value2 in satisfied['Concrete Vlaues2']:
                                                    for refinment in sorted_possible_refinments1:
                                                        if concrete_value1 == refinment['refinements'][0]['value']:
                                                            if concrete_value2 == refinment['refinements'][1]['value']:
                                                                Concrete_values.append({"condition1": concrete_value1, "condition2": concrete_value2, 
                                                                "Result": satisfied['Result'], "Similarity": refinment['distance']})

                                                                #if len(Concrete_values) == 40:
                                                                    #found = True
                                                                    #break


        end_time = time.time()
        Concrete_values_sorted = sorted(Concrete_values, key=lambda x: x['Similarity'])

        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions__Ranges_Partial.csv", index=False)

        satisfied_conditions_concrete_df = pd.DataFrame(Concrete_values_sorted)
        satisfied_conditions_concrete_df.to_csv("satisfied_conditions__Ranges_Concerete.csv", index=False)
        elapsed_time = end_time - start_time

        info = []
        refinement_info = {
            "Data Name": dataName,
            "Data Size": datasize,
            "type": "Ranges",
            "Access No.": counter,
            "Checked No.": check_counter,
            "Refinement No.": refinement_counter,
            "Time": round(elapsed_time, 3)
        }
        info.append(refinement_info)
        # Save info to CSV
        info_df = pd.DataFrame(info)
        info_df.to_csv("Run_info.csv", mode='a', index=False, header=False)

        print("Number of boxes access: ", counter+child_counter)
        print("Number of checked", check_counter)
        print("Number of refinments", refinement_counter)
        #print("Number of child clusters access: ", child_counter)
        #print("Number of Aggregation calculated: ", agg_counter)
        #print("Number of Ranges: ", ranges_counter)
        print("Time taken Overall for Partial filtered clusters:", round(elapsed_time, 3), "seconds") 
        


    def filter_clusters_partial_modified(self, cluster_key, parent_child_map, cluster_map, filtered_clusters, condition1, operator1, condition2, operator2, counter, child_counter, coverage_threshold=75):
        stack = [cluster_key]

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
                      
            # Partially satisfying
            elif self.applyOperator.apply_operator_ranges(cluster_info['Data_Min'][0],  cluster_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Partial") and self.applyOperator.apply_operator_ranges(
                cluster_info['Data_Min'][1], cluster_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Partial"): 
                
                if cluster_key in parent_child_map: 
                    child_list = []
                    for child_key in parent_child_map[cluster_key]:
                        child_counter+=1
                        child_info = cluster_map[child_key]

                        # Fully satisfying child
                        if self.applyOperator.apply_operator_ranges(child_info['Data_Min'][0],  child_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Full") and self.applyOperator.apply_operator_ranges(
                            child_info['Data_Min'][1], child_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Full"):
                            child_info['Satisfy'] = 'Full'
                            child_list.append({"satysfying": "full", "child": child_info})

                        # Partial child
                        elif self.applyOperator.apply_operator_ranges(child_info['Data_Min'][0],  child_info['Data_Max'][0], cond1_min, cond1_max, operator1, "Partial") and self.applyOperator.apply_operator_ranges(
                            child_info['Data_Min'][1], child_info['Data_Max'][1], cond2_min, cond2_max, operator2, "Partial"): 
                            child_info['Satisfy'] = 'Partial'

                            child_list.append({"satysfying": "partial", "child": [child_key]})


                    for child in child_list:
                            if child['satysfying'] == "partial":
                                stack.extend(child["child"])

                            elif child['satysfying'] == "full":
                                filtered_clusters.append(child["child"])
                      
                else:
                    cluster_info['Satisfy'] = 'Partial'
                    filtered_clusters.append(cluster_info)
                      
                   
            #Not satisfying cluster   
            else:
                # If the cluster does not satisfy the condition, we explore its children if any
                stack.extend(parent_child_map.get(cluster_key, []))
        
        return filtered_clusters, counter, child_counter
