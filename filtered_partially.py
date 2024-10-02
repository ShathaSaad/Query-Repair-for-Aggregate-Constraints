import timeit
import pandas as pd
from collections import defaultdict
import time
from constraint_evaluation import constraint_evaluation
from constraint_evaluation1 import constraint_evaluation1

from operators import operators
import bisect
import time
from filtered_fully import filtered_fully



class filtered_partially:
    def __init__(self):
        self.applyOperator = operators()
        self.filter_fully =  filtered_fully()
        self.evaluate_constraint = constraint_evaluation()
        self.evaluate_constraint1 = constraint_evaluation1()

    def check_predicates_partial_modified(self, sorted_possible_refinements, statistical_tree, expression, datasize, dataName):
        counter = 0
        agg_counter = 0
        child_counter = 0
        satisfied_conditions = []
        found = False

        check_counter = 0
        refinement_counter = 0
        
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
            # Loop over each column in the refinement dictionary
            condition1_operator = refinement['refinements'][0]['operator']
            condition1 = refinement['refinements'][0]['value'] 
            condition2_operator = refinement['refinements'][1]['operator']
            condition2 = refinement['refinements'][1]['value'] 
            #print(condition1_operator, condition1, condition2_operator, condition2)
            similarity = refinement['distance']

            filtered_clusters = []
            for root_key in root_clusters:
                filtered_clusters, counter, child_counter = self.filter_clusters_partial_modified(root_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, condition2_operator, filtered_clusters, counter, child_counter)
        
            # Convert the filtered clusters to a DataFrame and save to CSV
            filtered_df = pd.DataFrame(filtered_clusters)

            # Create a filename based on the conditions
            check_counter+=1
            satisfied, agg_counter = self.evaluate_constraint1.calculate_expression_partially(filtered_df, condition1, condition2, agg_counter, expression, "partial", similarity)
            #satisfied, agg_counter = evaluate_constraint.cardinality(filtered_df, counter, agg_counter, condition1, condition2)
            
            if satisfied != []:
                if satisfied['Range Satisfaction'] == 'Full':
                    refinement_counter+=1
                    satisfied_conditions.append(satisfied)
                    #if len(satisfied_conditions) == 7:
                        #break
                elif satisfied['Range Satisfaction'] == 'Partial':
                    filtered_clusters = []
                    check_counter+=1
                    for root_key in root_clusters:
                        filtered_clusters, counter = self.filter_fully.filter_clusters_Hash(root_key, cluster_map, condition1, condition1_operator, condition2, condition2_operator, filtered_clusters, counter, parent_child_map)
                       
                    filtered_df = pd.DataFrame(filtered_clusters)
                    satisfied, agg_counter = self.evaluate_constraint.evaluate_constraint(filtered_df, expression, condition1, condition2, agg_counter, similarity, "full")
                    
                    #satisfied, agg_counter = evaluate_constraint.cardinality(filtered_df, counter, agg_counter, condition1, condition2)
                    if satisfied != []:
                        refinement_counter+=1
                        satisfied_conditions.append(satisfied)
                        #if len(satisfied_conditions) == 7:
                            #break

            #filename = f'filtered_Partially_income_{condition1}_children_{condition2}_refined.csv'
            #filename = filename.replace(" ", "_").replace(",", "_")
            # Save to CSV
            #filtered_df.to_csv(filename, index=False)
        
        end_time = time.time() 
        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions_partial_Modified.csv", index=False)
        elapsed_time = end_time - start_time
        
        info = []
        refinement_info = {
            "Data Name": dataName,
            "Data Size": datasize,
            "Access No.": counter,
            "Checked No.": check_counter,
            "Refinement No.": refinement_counter,
            "Time": round(elapsed_time, 3)
        }
        info.append(refinement_info)
        # Save info to CSV
        info_df = pd.DataFrame(info)
        info_df.to_csv("Fully_info.csv", mode='a', index=False, header=False)

        print("Number of boxes access: ", counter+child_counter)
        print("Number of checked", check_counter)
        print("Number of refinments", refinement_counter)
        print("Time taken Overall:", round(elapsed_time, 3), "seconds") 
        #print("Number of child boxes checked: ", child_counter)
        #print("Number of Aggregation calculated: ", agg_counter)
        


    def filter_clusters_partial_modified(self, cluster_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, condition2_operator, filtered_clusters, counter, child_counter, coverage_threshold=75):
        stack = [cluster_key]
        lower_bound = 0
        upper_bound = cluster_map[cluster_key]['Count'] 

        while stack:
            counter = counter + 1
            cluster_key = stack.pop()
            cluster_info = cluster_map[cluster_key]
        
            # Extract relevant data
            data_Count = cluster_info['Count']
            data_Min = cluster_info['Data_Min']
            data_Max = cluster_info['Data_Max']

            # Check if this cluster fully satisfies the conditions
            if (self.applyOperator.apply_operator(data_Min[0], data_Max[0], condition1, condition1_operator, "Full") and 
                self.applyOperator.apply_operator(data_Min[1], data_Max[1], condition2, condition2_operator, "Full")):
                cluster_info['Satisfy'] = 'Full' 
                filtered_clusters.append(cluster_info)
                lower_bound += data_Count
                
            elif (self.applyOperator.apply_operator(data_Min[0], data_Max[0], condition1, condition1_operator, "Partial") and 
                self.applyOperator.apply_operator(data_Min[1], data_Max[1], condition2, condition2_operator, "Partial")):
                if cluster_key in parent_child_map:
                    not_satysfying_count = 0
                    child_list = []

                    for child_key in parent_child_map[cluster_key]:
                        child_counter+=1
                        child_info = cluster_map[child_key]
                        child_count = child_info['Count']
                        child_Min = child_info['Data_Min']
                        child_Max = child_info['Data_Max']

                        # Fully satisfying child
                        if (self.applyOperator.apply_operator(child_Min[0], child_Max[0], condition1, condition1_operator, "Full") and 
                            self.applyOperator.apply_operator(child_Min[1], child_Max[1], condition2, condition2_operator, "Full")):
                            lower_bound += child_count
                            child_info['Satisfy'] = 'Full'
                            child_list.append({"satysfying": "full", "child": child_info})

                        # Partial child
                        elif (self.applyOperator.apply_operator(child_Min[0], child_Max[0], condition1, condition1_operator, "Partial") and 
                            self.applyOperator.apply_operator(child_Min[1], child_Max[1], condition2, condition2_operator, "Partial")):
                            child_list.append({"satysfying": "partial", "child": [child_key]})
                        
                        # Not satisfying child
                        else:
                            upper_bound-=child_count
                            not_satysfying_count+=child_count

                    satisfaction_percentage = (lower_bound / upper_bound) * 100 if upper_bound > 0 else 0
                    
                    if satisfaction_percentage >= coverage_threshold and not_satysfying_count < data_Count/2: #not_satysfying_count/data_Count*100 <= 100-coverage_threshold:   
                        # The partial cluster satisfies the threshold
                        cluster_info['Satisfy'] = 'Partial'
                        filtered_clusters.append(cluster_info)
                        
                    else:
                        for child in child_list:
                                if child['satysfying'] == "partial":
                                    stack.extend(child["child"])
                                elif child['satysfying'] == "full":
                                    filtered_clusters.append(child["child"])
            #Not satisfying cluster    
            else:
                upper_bound -= data_Count

                # If the cluster does not satisfy the condition, we explore its children if any
                if cluster_key in parent_child_map:
                    stack.extend(parent_child_map[cluster_key])                        
        
        return filtered_clusters, counter, child_counter

    
    def check_predicates_partial_inc(self, sorted_possible_refinements, statistical_tree, descendants, expression):
        counter = 0
        agg_counter = 0
        child_counter = 0
        filtered_clusters_list = []
        satisfied_conditions = []
        filtered_clusters_dict = {}
        find_cluster_time = 0
        filter_remainder_time = 0
        intersection_time = 0
        partial_time = 0
        incremental_time = 0

        # Create dictionaries for parent-child relationships and cluster info
        parent_child_map = defaultdict(list)
        cluster_map = {}

        for row in statistical_tree:
            parent_key = (row['Parent level'], row['Parent cluster'])
            child_key = (row['Level'], row['Cluster Id'])
            parent_child_map[parent_key].append(child_key)
            cluster_map[child_key] = row

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
                    start0 = time.time()
                    filtered_clusters, counter, child_counter = self.filter_clusters_partial_modified(
                        root_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, 
                        condition2_operator, filtered_clusters, counter, child_counter)
                    key = (condition1, condition2)
                    filtered_clusters_dict[key] = filtered_clusters
                    end0 = time.time()
                    partial_time += (end0 - start0)
            else:
                start1 = time.time()
                # If both conditions are at index > 0, use intersection
                filtered_clusters, find_cluster_time, intersection_time, filter_remainder_time = self.filter_by_intersection_and_remainder(find_cluster_time, intersection_time, filter_remainder_time, filtered_clusters_dict, sorted_possible_refinements,
                    cluster_map, condition1, condition1_operator, condition2, condition2_operator, 
                    filtered_clusters_list, parent_child_map, counter,child_counter, descendants)
                
                key = (condition1, condition2)
                filtered_clusters_dict[key] = filtered_clusters
                end1 = time.time()
                incremental_time += (end1 - start1)

            filtered_df = pd.DataFrame(filtered_clusters)

            evaluate_constraint1 = constraint_evaluation1()
            satisfied, agg_counter = evaluate_constraint1.calculate_expression_partially(filtered_df, condition1, condition2, agg_counter, expression, "partial", " ")
            
            if satisfied:
                satisfied_conditions.append(satisfied)

            #filename = f'filtered_partial_income_{condition1}_children_{condition2}_Incremental.csv'
            #filename = filename.replace(" ", "_").replace(",", "_")
            #filtered_df.to_csv(filename, index=False)
        
        end_time = time.time()
        
        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions_partial_Incremental.csv", index=False)


        print("Number of boxes checked:", counter)
        print("Number of child boxes checked: ", child_counter)
        print("Number of Aggregation calculated:", agg_counter)
        elapsed_time = end_time - start_time
        print("Time taken Overall for Fully filtered clusters:", round(elapsed_time, 3), "seconds")
        print("Time details:", "partial_time:", round(partial_time, 3), "incremental_time:", round(incremental_time, 3), "find_prev_cluster_time:", round(find_cluster_time, 3) , 
            "intersection_time:", round(intersection_time, 3), "intersection_time:", round(filter_remainder_time, 3), "seconds")

    def find_immediate_previous(self, sorted_values, current_value):
        # Ensure the values are sorted and unique when they are first initialized or updated
        idx = bisect.bisect_left(sorted_values, current_value)
        
        # If idx is more than 0, then current_value's immediate predecessor is at idx-1
        if idx > 0:
            return sorted_values[idx-1]
        else:
            # If idx is 0, current_value is the smallest or not in the list at all
            return None


    def filter_by_intersection_and_remainder(self, find_cluster_time, intersection_time, filter_remainder_time, filtered_clusters_dict, sorted_possible_refinements, cluster_map, condition1, condition1_operator, condition2, condition2_operator, filtered_clusters_list, parent_child_map, counter, child_counter, descendants):
        start1 = time.time()
        condition1_values = [item[0]['value'] for item in sorted_possible_refinements]
        condition2_values = [item[1]['value'] for item in sorted_possible_refinements]
        
        # Find immediate previous values
        prev_condition1 = self.find_immediate_previous(condition1_values, condition1)
        prev_condition2 = self.find_immediate_previous(condition2_values, condition2)

        # Prepare the previous cluster keys
        prev1_cluster_key = (condition1, prev_condition2) 
        prev2_cluster_key = (prev_condition1, condition2) 

        # Retrieve clusters from the map if keys are valid
        prev1_cluster = filtered_clusters_dict.get(prev1_cluster_key) 
        prev2_cluster = filtered_clusters_dict.get(prev2_cluster_key) 

        end1 = time.time()
        start2 = time.time ()
        prev_filtered_1_dict = { (c['Level'], c['Cluster Id']): c for c in prev1_cluster }
        prev_filtered_2_dict = { (c['Level'], c['Cluster Id']): c for c in prev2_cluster }

        intersection_keys = set(prev_filtered_1_dict.keys()).intersection(set(prev_filtered_2_dict.keys()))
        intersection_clusters = []

        for key in intersection_keys:
            cluster1 = prev_filtered_1_dict[key]
            cluster2 = prev_filtered_2_dict[key]
            if cluster1['Satisfy'] == 'Full' and cluster2['Satisfy'] == 'Full':
                cluster1['Satisfy'] = 'Full'  # Both clusters are fully satisfied
            else:
                cluster1['Satisfy'] = 'Partial'  # Partial satisfaction
            intersection_clusters.append(cluster1)

        remainder_1_keys = set(prev_filtered_1_dict.keys()).difference(intersection_keys)
        remainder_2_keys = set(prev_filtered_2_dict.keys()).difference(intersection_keys)

        remainder_1 = [prev_filtered_1_dict[k] for k in remainder_1_keys]
        remainder_2 = [prev_filtered_2_dict[k] for k in remainder_2_keys]
        remainder = remainder_1 + remainder_2
        end2 = time.time()
        start3 = time.time()
        # Filter these remainders against the current conditions   
        filtered_remainder, counter = self.filter_remainder(parent_child_map, cluster_map, remainder, intersection_clusters, condition1, condition1_operator, condition2, condition2_operator, counter, descendants)
        
        # Combine the results
        combined_filtered_clusters = intersection_clusters + filtered_remainder # + filtered_remainder_2
        unique_combined_filtered_clusters = { (c['Level'], c['Cluster Id']): c for c in combined_filtered_clusters }
        end3 = time.time()
        find_cluster_time += (end1 - start1)
        intersection_time += ( end2 - start2)
        filter_remainder_time +=(end3 - start3)

        return list(unique_combined_filtered_clusters.values()), find_cluster_time, find_cluster_time, filter_remainder_time

    def filter_remainder(self, parent_child_map, cluster_map, remainder_clusters, intersection_clusters, condition1, condition1_operator, condition2, condition2_operator, counter, descendants):
        new_clusters = intersection_clusters

        # sort by 'Count' descending
        remainder_clusters_sorted = sorted(remainder_clusters, key=lambda x: x['Count'], reverse=True)
        
        for cluster_info in remainder_clusters_sorted:
            counter = counter+1
            cluster_key = (cluster_info['Level'], cluster_info['Cluster Id'])
            cluster_parent = (cluster_info['Parent level'], cluster_info['Parent cluster'])
            should_add = True

            # Combine intersection_clusters with newly added clusters
            #combined_clusters = intersection_clusters + new_clusters

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
                    filtered_clusters_partial, counter = self.filter_clusters_Hash(cluster_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, condition2_operator, counter)
                    new_clusters.extend(filtered_clusters_partial)

        return new_clusters, counter

    def filter_clusters_Hash(self, current_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, condition2_operator, counter):
        stack = [current_key]
        filtered_clusters = []

        while stack:
            counter += 1
            current_key = stack.pop()
            current_cluster = cluster_map[current_key]

            if (self.applyOperator.apply_operator(current_cluster['Data_Min'][0], current_cluster['Data_Max'][0], condition1, condition1_operator, "Full") and 
                self.applyOperator.apply_operator(current_cluster['Data_Min'][1], current_cluster['Data_Max'][1], condition2, condition2_operator, "Full")):
                current_cluster['Satisfy'] = 'Full'
                filtered_clusters.append(current_cluster)
            
            # Otherwise, check the child clusters
            else:
                stack.extend(parent_child_map.get(current_key, []))

        return filtered_clusters, counter

    def get_all_descendants(self, cluster_key_check, parent_child_map):

        descendants = set()
        children = parent_child_map.get(cluster_key_check, [])

        # Iterate through each child
        for child in children:
            # Add the immediate child
            descendants.add(child)
            # Recursively add all descendants of this child
            descendants.update(self.get_all_descendants(child, parent_child_map))
    
        return descendants

    '''   
    def filter_remainder(self, remainder_clusters, condition1, condition1_operator, condition2, condition2_operator, parent_child_map, cluster_map, counter):
        filtered_clusters_partial = []
        filtered_clusters_list = []   
        applyOperator = operators()
    

        for cluster_info in remainder_clusters:
            counter = counter+1
            data_Min = cluster_info['Data_Min']
            data_Max = cluster_info['Data_Max']
            
            cluster_key = (cluster_info['Level'], cluster_info['Cluster Id'])
            #if (cluster_info['Data_Min'][0] >= condition1) and (cluster_info['Data_Min'][1] >= condition2):
            if (applyOperator.apply_operator(data_Min[0], data_Max[0], condition1, condition1_operator) and applyOperator.apply_operator(data_Min[1], data_Max[1], condition2, condition2_operator)):
                filtered_clusters_list.append(cluster_info)
            elif cluster_info['Count'] > 1:
                filtered_clusters_partial, counter = self.filter_clusters_Hash(cluster_key, parent_child_map, cluster_map, condition1, condition1_operator, condition2, condition2_operator, counter)
                
                filtered_clusters_list += filtered_clusters_partial

        return filtered_clusters_list, counter
    ''' 

    '''
    def check_predicates_partial(self, sorted_possible_refinements, statistical_tree):
        counter = 0
        agg_counter = 0
        child_counter = 0
        satisfied_conditions = []

        df_statistical_info = pd.DataFrame(statistical_tree)

        # Create a dictionary to hold parent-child relationships
        parent_child_map = defaultdict(list)
        # Create a dictionary to hold cluster information by their keys for quick access
        cluster_map = {}

        for _, row in df_statistical_info.iterrows():
            parent_key = (row['Parent level'], row['Parent cluster'])
            child_key = (row['Level'], row['Cluster Id'])
            parent_child_map[parent_key].append(child_key) 
            cluster_map[child_key] = row
        

        # Find the root clusters (clusters with Parent level 0)
        root_clusters = df_statistical_info[(df_statistical_info['Parent level'] == 0)]

        start_time = time.time() 

        
        for refinement in sorted_possible_refinements:
            condition1 = refinement[0]
            condition2 = refinement[1]   
            condition3 = refinement[2]          
        
            filtered_clusters = []
            for _, root in root_clusters.iterrows():
                root_key = (root['Level'], root['Cluster Id'])
                filtered_clusters, counter, child_counter = self.filter_clusters_partial_deep(root_key, parent_child_map, df_statistical_info, condition1, condition2, condition3, filtered_clusters, counter, child_counter, coverage_threshold = 75)
        

            # Convert the filtered clusters to a DataFrame and save to CSV
            filtered_df = pd.DataFrame(filtered_clusters)

            # Create a filename based on the conditions
            evaluate_constraint = constraint_evaluation()
            satisfied, agg_counter = evaluate_constraint.calculate_spd_partially(filtered_df, counter, agg_counter, child_counter, condition1, condition2, condition3)
            if satisfied != []:
                    satisfied_conditions.append(satisfied)
            
            #satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
            #satisfied_conditions_df.to_csv("satisfied_conditions_Partial_12.csv", index=False)
            
            #filename = f'filtered_Partially_income_{condition1}_children_{condition2}_12.csv'
            #filename = filename.replace(" ", "_").replace(",", "_")
            # Save to CSV
            #filtered_df.to_csv(filename, index=False)
            #print(f"Filtered results saved to {filename}")

        end_time = time.time() 
        print("Number of boxes checked: ", counter)
        print("Number of child boxes checked: ", child_counter)
        print("Number of Aggregation calculated: ", agg_counter)
        elapsed_time = end_time - start_time
        print("Time taken Overall for Partial filtered clusters:", round(elapsed_time, 3), "seconds") 


    def filter_clusters_partial_deep(self, cluster_key, cluster_map, data, condition1, condition2, filtered_clusters, counter, child_counter, coverage_threshold):
        stack = [cluster_key]

        while stack:
            counter = counter +1
            cluster_key = stack.pop()
            level, cluster_id = cluster_key
            cluster_info = data[(data['Level'] == level) & (data['Cluster Id'] == cluster_id)].iloc[0]

            data_Min = cluster_info['Data_Min']
            data_Count = cluster_info['Count']

            # Convert cluster info to dict
            cluster_dict = cluster_info.to_dict()

            if data_Min[0] >= condition1 and data_Min[1] >= condition2:
                # Fully satisfying cluster
                cluster_dict['Satisfy'] = 'Full'
                filtered_clusters.append(cluster_dict)

                #print(f"Cluster {cluster_key} is fully satisfying")
            else:
                # Calculate partial satisfaction by examining children
                children = cluster_map.get(cluster_key, [])
                if children:
                    satisfied_points = 0
                    stack_partial = children.copy()

                    while stack_partial and satisfied_points / data_Count * 100 < coverage_threshold:
                        child_key = stack_partial.pop()
                        child_counter = child_counter +1
                        level1, cluster_id1 = child_key

                        child_info = data[(data['Level'] == level1) & (data['Cluster Id'] == cluster_id1)].iloc[0]
                        child_data_Min = child_info['Data_Min']
                        child_count = child_info['Count']

                        if child_data_Min[0] >= condition1 and child_data_Min[1] >= condition2:
                            # Child is fully satisfying, add its count
                            satisfied_points += child_count
                            satisfaction_percentage = (satisfied_points / data_Count) * 100

                            if satisfaction_percentage >= coverage_threshold:
                                cluster_dict['Satisfy'] = 'Partial'
                                filtered_clusters.append(cluster_dict)
                                #print(f"Cluster {cluster_key} is partially satisfying with sufficient coverage")
                                break  # Stop further exploration for this cluster

                        else:
                            # Explore deeper if the child is not fully satisfying
                            grand_children = cluster_map.get(child_key, [])
                            stack_partial.extend(grand_children)

                    # If not enough satisfaction found, continue exploring other clusters
                    if satisfied_points / data_Count * 100 < coverage_threshold:
                        stack.extend(children)

                else:
                    continue

        return filtered_clusters, counter, child_counter            
        '''       

  