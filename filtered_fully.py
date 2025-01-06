import pandas as pd
import numpy as np
import time
from collections import defaultdict
from constraint_evaluation import constraint_evaluation
from operators import operators
import bisect
import os



class filtered_fully:
    def __init__(self):
        self.applyOperator = operators()

    def check_predicates(self, sorted_possible_refinements, statistical_tree, expression, datasize, dataName, result_num, query_num, constraint, predicates, combination, distribution, Correlation, constraint_type):
        counter, i = 0, 0
        agg_counter = 0
        satisfied_conditions = []
        check_counter_Not = 0
        check_counter = 0
        refinement_counter = 0
        solutions_count = 0
        end_time = 0
        checked_num = 0
        accesses_num = 0
        refinment_num = 0

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
        
        for refinement in sorted_possible_refinements:
            # Loop over each condition in the refinement dictionary   
            conditions = []
            operators = []
            for ref in refinement['refinements']:
                conditions.append(ref['value'])
                operators.append(ref['operator'])

            #print(conditions)

            similarity = refinement['distance']

            filtered_clusters = []          
            for root_key in root_clusters:
                filtered_clusters, counter = self.filter_clusters_Hash(root_key, cluster_map, conditions, operators, filtered_clusters, counter, parent_child_map)
                
            # Convert the filtered clusters to a DataFrame and save to CSV
            filtered_df = pd.DataFrame(filtered_clusters)

            # Evaluate and store the results
            evaluate_constraint = constraint_evaluation() 
            check_counter += 1
            #satisfied, agg_counter = evaluate_constraint.cardinality(filtered_df, counter, agg_counter, condition1, condition2)
            satisfied, agg_counter, Not_satisfied, result = evaluate_constraint.evaluate_constraint1(filtered_df, expression, conditions, agg_counter, similarity, "full", constraint_type)

            if satisfied != []:
                #check_counter_satisfy += 1
                refinement_counter += 1
                satisfied_conditions.append(satisfied)
                solutions_count +=1


                if len(satisfied_conditions) == result_num:
                    end_time = time.time() 
                    checked_num = check_counter
                    accesses_num = counter
                    refinment_num = refinement_counter
                    break
        
        if end_time == 0: #no solutions
            end_time = time.time()
            checked_num = check_counter
            accesses_num = counter
            refinment_num = refinement_counter
                    
        self.print_results(satisfied_conditions, start_time, end_time, dataName, datasize, query_num, combination, 
                accesses_num, solutions_count, checked_num, check_counter_Not, refinment_num, constraint, predicates, distribution, result_num, Correlation)
            


    def print_results(self, satisfied_conditions, start_time, end_time, dataName, datasize, query_num, combination, 
        counter, solutions_count, check_counter, check_counter_Not, refinement_counter, constraint, predicates, distribution, result_num, Correlation):
                    
        '''
        directory = '/Users/Shatha/Downloads/Query_Refinment_Shatha/running_files'
        filename = f'filtered_fully_{conditions}.csv'
        filename = filename.replace(" ", "_").replace(",", "_")
        # Save to CSV
        # Join the directory and file name to create the full path
        full_path = os.path.join(directory, filename)
        filtered_df.to_csv(full_path, index=False)
        '''
        # Save satisfied conditions to CSV
        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        output_directory = "/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final1"
        # For the second file
        file_path_2 = os.path.join(
            output_directory, 
            f"satisfied_conditions_Fully_{dataName}_size{datasize}_query{query_num}_constraint{constraint}.csv"
        )
        satisfied_conditions_df.to_csv(file_path_2, index=False)

        elapsed_time = end_time - start_time

        info = []
        refinement_info = {
            "Data Name": dataName,
            "Data Size": datasize,
            "Query Num": query_num,
            "Type": "Fully",
            "Top-K": result_num,
            "Combinations Num": combination,
            "Distance": round((check_counter) / combination * 100, 3),
            "Access Num": counter,
            "Checked Num": (check_counter),
            "Refinement Num": refinement_counter,
            "Time": round(elapsed_time, 3),
            "Constraint Width": round(constraint[1]-constraint[0], 2),
            "Solutions Count": solutions_count,
            "Constraint": constraint, 
            #"Distribution": distribution,
            #"Correlation": Correlation,
            "Query": predicates,
            "Range Evaluation Time": " ",
            "Division Time": " ",
            "Single Time": " ",
            "Processing Time": " "
        }
        info.append(refinement_info)
        info_df = pd.DataFrame(info)
        output_directory = "/Users/Shatha/Downloads/Query_Refinment_Shatha/sh_Final1"
        # Ensure the directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Define the full file path including the directory
        file_path = os.path.join(output_directory, f"Run_info_{dataName}_size{datasize}_H.csv")

        write_header = not os.path.exists(file_path)  # Write header only if the file does not exist

        # Write to CSV with appropriate header setting
        info_df.to_csv(file_path, mode='a', index=False, header=write_header)

        print("\nTop-", result_num," :")
        print("Number of boxes access: ", counter)
        print("Number of checked", check_counter)
        #print("Checked No.-Not:", check_counter_Not)
        #print("Checked No.-Satisfy:", check_counter_satisfy)
        print("Number of refinments", refinement_counter)
        print("Time taken Overall:", round(elapsed_time, 3), "seconds")  


    def filter_clusters_Hash(self, current_key, cluster_map, conditions, operators, filtered_clusters, 
    counter, parent_child_map):
    
        stack = [current_key]
        conditions1 = enumerate(zip(conditions, operators))


        while stack:
            counter += 1
            current_key = stack.pop()
            current_cluster = cluster_map[current_key]

            # Initialize a flag to track if the current cluster satisfies all conditions
            fully_satisfies = True

            # Loop through each condition and operator
            for idx, (condition, operator) in conditions1:
                data_min = current_cluster['Data_Min'][idx]  # Get the corresponding Data_Min value for this condition
                data_max = current_cluster['Data_Max'][idx]  # Get the corresponding Data_Max value for this condition
                

                # If any condition is not satisfied, mark as False and break
                if not self.applyOperator.apply_operator(data_min, data_max, condition, operator, "Full"):
                    fully_satisfies = False
                    break

            # If all conditions are satisfied, mark the cluster as "Full"
            if fully_satisfies:
                current_cluster['Satisfy'] = 'Full'
                filtered_clusters.append(current_cluster)

            else:  # Otherwise, check the child clusters
                stack.extend(parent_child_map.get(current_key, []))

        return filtered_clusters, counter
    
    def filter_clusters_Hash1(self, current_key, cluster_map, condition1, condition1_operator, condition2, condition2_operator, condition3, condition3_operator, filtered_clusters, counter, parent_child_map):

        stack = [current_key]

        while stack:
            counter += 1
            current_key = stack.pop()
            current_cluster = cluster_map[current_key]

            if (self.applyOperator.apply_operator(current_cluster['Data_Min'][0], current_cluster['Data_Max'][0], condition1, condition1_operator, "Full") and 
                self.applyOperator.apply_operator(current_cluster['Data_Min'][1], current_cluster['Data_Max'][1], condition2, condition2_operator, "Full") and 
                self.applyOperator.apply_operator(current_cluster['Data_Min'][2], current_cluster['Data_Max'][2], condition3, condition3_operator, "Full")):
                current_cluster['Satisfy'] = 'Full'
                filtered_clusters.append(current_cluster)

                
            else: # Otherwise, check the child clusters
                stack.extend(parent_child_map.get(current_key, []))

        return filtered_clusters, counter

    def filter_clusters_Hash2(self, current_key, cluster_map, condition1, condition1_operator, condition2, condition2_operator, filtered_clusters, counter, parent_child_map):

        stack = [current_key]

        while stack:
            counter += 1
            current_key = stack.pop()
            current_cluster = cluster_map[current_key]

            if (self.applyOperator.apply_operator(current_cluster['Data_Min'][0], current_cluster['Data_Max'][0], condition1, condition1_operator, "Full") and 
                self.applyOperator.apply_operator(current_cluster['Data_Min'][1], current_cluster['Data_Max'][1], condition2, condition2_operator, "Full")):
                current_cluster['Satisfy'] = 'Full'
                filtered_clusters.append(current_cluster)

                
            else: # Otherwise, check the child clusters
                stack.extend(parent_child_map.get(current_key, []))

        return filtered_clusters, counter
            
    def check_predicates_inc(self, sorted_possible_refinements, statistical_tree, descendants, expression):
        counter = 0
        agg_counter = 0
        satisfied_conditions = []
        filtered_clusters_dict = {}
        #find_cluster_time = 0
        #filter_remainder_time = 0

        # Create dictionaries for parent-child relationships and cluster info
        parent_child_map = defaultdict(list)
        cluster_map = {}

        for row in statistical_tree:
            parent_key = (row['Parent level'], row['Parent cluster'])
            child_key = (row['Level'], row['Cluster Id'])
            parent_child_map[parent_key].append(child_key) 
            cluster_map[child_key] = row  

        condition1_values = [item[0]['value'] for item in sorted_possible_refinements]
        condition2_values = [item[1]['value'] for item in sorted_possible_refinements]

        root_clusters = [(row['Level'], row['Cluster Id']) for row in statistical_tree if row['Parent level'] == 0]

        start_time = time.time()

        for idx, refinement in enumerate(sorted_possible_refinements):
            # Loop over each column in the refinement dictionary
            condition1_operator = refinement[0]['operator']
            condition1 = refinement[0]['value']
            condition2_operator = refinement[1]['operator']
            condition2 = refinement[1]['value'] 

            # Check if this is the first refinement or if condition1 or condition2 matches the first value in the list
            if idx == 0 or condition1 == sorted_possible_refinements[0][0]['value'] or condition2 == sorted_possible_refinements[0][1]['value']:
                # If it's the first condition or matches the first value in the list, filter the entire tree

                filtered_clusters = []
                for root_key in root_clusters:
                    filtered_clusters, counter = self.filter_clusters_Hash(root_key, cluster_map, condition1, condition1_operator, 
                    condition2, condition2_operator, filtered_clusters, counter, parent_child_map)

                    key = (condition1, condition2)
                    filtered_clusters_dict[key] = filtered_clusters
            else:
                # If both conditions are at index > 0, use intersection
                filtered_clusters  = self.filter_by_intersection_and_remainder(filtered_clusters_dict, parent_child_map, cluster_map, condition1, 
                condition1_operator, condition2, condition2_operator, counter, descendants, condition1_values, condition2_values)

                key = (condition1, condition2)
                filtered_clusters_dict[key] = filtered_clusters

            filtered_df = pd.DataFrame(filtered_clusters)

            evaluate_constraint = constraint_evaluation()
            satisfied, agg_counter = evaluate_constraint.evaluate_constraint(filtered_df, expression, condition1, condition2, agg_counter, " ")
            if satisfied:
                satisfied_conditions.append(satisfied)

            #filename = f'filtered_fully_income_{condition1}_children_{condition2}_Incremental.csv'
            #filename = filename.replace(" ", "_").replace(",", "_")
            #filtered_df.to_csv(filename, index=False)

        end_time = time.time()

        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions_Incremental.csv", index=False)

        print("Number of boxes checked:", counter)
        print("Number of Aggregation calculated:", agg_counter)
        elapsed_time = end_time - start_time
        print("Time taken Overall for Fully filtered clusters:", round(elapsed_time, 3), "seconds")
        #print("Time taken Overall to find previous clusters:", round(find_cluster_time, 3), "seconds") 
        #print("Time taken Overall to filter remainder clusters:", round(filter_remainder_time, 3), "seconds")
        #print("Time taken Overall to find descendants clusters:", round(find_descendants_time, 3), "seconds")

    def find_immediate_previous(self, sorted_values, current_value):
        # Ensure the values are sorted and unique when they are first initialized or updated
        idx = bisect.bisect_left(sorted_values, current_value)
        
        # If idx is more than 0, then current_value's immediate predecessor is at idx-1
        if idx > 0:
            return sorted_values[idx-1]
        else:
            # If idx is 0, current_value is the smallest or not in the list at all
            return None


    def filter_by_intersection_and_remainder(self, filtered_clusters_dict, parent_child_map, cluster_map, condition1, 
    condition1_operator, condition2, condition2_operator, counter, descendants, condition1_values, condition2_values):
        #start1 = time.time()
        
        # Find immediate previous values
        prev_condition1 = self.find_immediate_previous(condition1_values, condition1)
        prev_condition2 = self.find_immediate_previous(condition2_values, condition2)

        # Prepare the previous cluster keys
        prev1_cluster_key = (condition1, prev_condition2) 
        prev2_cluster_key = (prev_condition1, condition2) 

        # Retrieve clusters from the map if keys are valid
        prev1_cluster = filtered_clusters_dict.get(prev1_cluster_key) 
        prev2_cluster = filtered_clusters_dict.get(prev2_cluster_key) 

        prev_filtered_1_dict = { (c['Level'], c['Cluster Id']): c for c in prev1_cluster }
        prev_filtered_2_dict = { (c['Level'], c['Cluster Id']): c for c in prev2_cluster }
        #end1 = time.time()
        #comb1 = end1 - start1        
        #find_cluster_time += comb1
  
        intersection_keys = set(prev_filtered_1_dict.keys()).intersection(set(prev_filtered_2_dict.keys()))
        intersection_clusters = [prev_filtered_1_dict[k] for k in intersection_keys]

        remainder_1_keys = set(prev_filtered_1_dict.keys()).difference(intersection_keys)
        remainder_2_keys = set(prev_filtered_2_dict.keys()).difference(intersection_keys)

        remainder_1 = [prev_filtered_1_dict[k] for k in remainder_1_keys]
        remainder_2 = [prev_filtered_2_dict[k] for k in remainder_2_keys]
        remainder = remainder_1 + remainder_2
        #start2 = time.time()
        # Filter these remainders against the current conditions
        filtered_remainder, counter= self.filter_remainder(parent_child_map, cluster_map, remainder, 
        intersection_clusters, condition1, condition1_operator, condition2, condition2_operator, counter, descendants)
        #end2 = time.time()
        #comb2 = end2 - start2       
        #filter_remainder_time += comb2

        # Combine the results
        combined_filtered_clusters = intersection_clusters + filtered_remainder
        unique_combined_filtered_clusters = { (c['Level'], c['Cluster Id']): c for c in combined_filtered_clusters }

        return list(unique_combined_filtered_clusters.values())
        
    
    def filter_remainder(self, parent_child_map, cluster_map, remainder_clusters, intersection_clusters, condition1, 
    condition1_operator, condition2, condition2_operator, counter, descendants):
        filtered_partial = []
        new_clusters = intersection_clusters

        # sort by 'Count' descending
        remainder_clusters_sorted = sorted(remainder_clusters, key=lambda x: x['Count'], reverse=True)
        
        for cluster_info in remainder_clusters_sorted:
            counter = counter+1
            cluster_key = (cluster_info['Level'], cluster_info['Cluster Id'])
            cluster_parent = (cluster_info['Parent level'], cluster_info['Parent cluster'])
            should_add = True
            
            for item in new_clusters:

                # Check if cluster_info is not a child of any cluster in filtered_clusters
                cluster_key_check = (item['Level'], item['Cluster Id'])
                
                if (cluster_parent == cluster_key_check or 
                    cluster_parent in descendants.get(cluster_key_check, set())):
                    should_add = False
                    break
            
            # Process further only if should_add is True
            if should_add:
                if (self.applyOperator.apply_operator(cluster_info['Data_Min'][0], cluster_info['Data_Max'][0], condition1, condition1_operator, "Full") and 
                    self.applyOperator.apply_operator(cluster_info['Data_Min'][1], cluster_info['Data_Max'][1], condition2, condition2_operator, "Full")):  
                    new_clusters.append(cluster_info)
                    
                elif cluster_info['Count'] > 1:
                    
                    filtered_clusters_partial, counter = self.filter_clusters_Hash(cluster_key, cluster_map, condition1, condition1_operator, 
                    condition2, condition2_operator, filtered_partial, counter, parent_child_map)
                    new_clusters.extend(filtered_clusters_partial)

        return new_clusters, counter



    '''   
    def filter_clusters1(self, cluster_key, parent_child_map, cluster_map, data, condition1, condition2, filtered_clusters, counter):
        stack = [cluster_key]
        
        while stack:
            counter = counter +1
            cluster_key = stack.pop()
            level, cluster_id = cluster_key
            cluster_info = data[(data['Level'] == level) & (data['Cluster Id'] == cluster_id)].iloc[0]
    
             # Access min data values
            data_Min = cluster_info['Data_Min']
            
            all_points_satisfy = (data_Min[0] >= condition1) and (data_Min[1] >= condition2)

            if all_points_satisfy:
                filtered_clusters.append(cluster_info.to_dict())

            else:
                children = parent_child_map.get(cluster_key, [] )
                if children:
                    stack.extend(children)

        return filtered_clusters, counter
    
    def filter_clusters2(self, cluster_key, parent_child_map, cluster_map, condition1, condition2, filtered_clusters, counter, data):
        stack = [cluster_key]
        

        while stack:
            counter = counter +1
            cluster_key = stack.pop()
            level, cluster_id = cluster_key
            cluster_info = data[(data['Level'] == level) & (data['Cluster Id'] == cluster_id)].iloc[0]
    
             # Access min data values
            data_Min = cluster_info['Data_Min']

            # Convert cluster info to dict
            cluster_dict = cluster_info.to_dict()
            
            all_points_satisfy = (data_Min[0] >= condition1) and (data_Min[1] >= condition2)

            if all_points_satisfy:
                cluster_dict['Satisfy'] = 'Full'
                filtered_clusters.append(cluster_dict)

            else:
                children = cluster_map.get(cluster_key, [] )

                if children:
                    for child_key in children:
                        child_info = data[(data['Level'] == child_key[0]) & (data['Cluster Id'] == child_key[1])].iloc[0]
                        child_data_Min = child_info['Data_Min']
                        child_data_Max = child_info['Data_Max']
                        child_dict = child_info.to_dict()

                        if child_data_Min[0] >= condition1 and child_data_Min[1] >= condition2:
                            filtered_clusters.append(child_dict)

                        elif child_data_Max[0] < condition1 or child_data_Max[1] < condition2:
                            continue

                        else:
                            stack.extend([child_key])

        return filtered_clusters, counter
        '''
