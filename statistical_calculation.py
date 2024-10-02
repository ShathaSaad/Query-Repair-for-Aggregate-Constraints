import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import re


class statistical_calculation:
    def __init__(self):
        self.column_set = set()

    def evaluate_aggregation(self, df, expression):
        # Helper function to parse and evaluate conditions
        def eval_cond(df, cond):
            cond = cond.replace('"', '')  # Remove quotes for direct column access
            
            # Attempt to find all column names using a simple regex pattern
            columns = re.findall(r'\b\w+\b(?=\s*==|\s*>=|\s*<=|\s*!=|\s*<|\s*>)', cond)
            unique_columns = list(set(columns))  # Remove duplicates
            self.column_set.update(unique_columns)  # Update the class-wide column set

            return df.query(cond), unique_columns

        # Define aggregation mappings
        aggregation_functions = {
            'count': lambda x: eval_cond(df, x)[0].shape[0],
            'sum': lambda x, col: eval_cond(df, x)[0][col].sum(),
            'mean': lambda x, col: eval_cond(df, x)[0][col].mean(),
        }
        
        # Parse the expression to find which aggregation to apply
        parts = expression.split('(')
        func_name = parts[0].strip()
        args = parts[1].strip()[:-1]  # remove trailing ")"
        args = [arg.strip() for arg in args.split(',')]

        if len(args) == 1:
            result = aggregation_functions[func_name](args[0])
        elif len(args) == 2:
            args[1] = args[1].replace('"', '').strip()  # clean up column name
            result = aggregation_functions[func_name](*args)

        return result

    def statistical_calculation(self, cluster_tree, df, aggregations, predicates_number, constraint_columns):
        statistical_tree =[]

        for clusters in cluster_tree:
            data_points_array = np.array(clusters['Data points'])   
            
            sliced_data_points_array = data_points_array[:, predicates_number:] # Slice the array starting from predicates_number to get constraint points

            df_sliced = pd.DataFrame(sliced_data_points_array)  # Create a DataFrame from the sliced array

            # Assign constraints' column names
            df_sliced.columns = constraint_columns
            
            box = self.points_bounds(data_points_array)
            count = self.count(len(clusters['Data points']))
            #cmplx_constraint_SPD = self.complex_constraint_SPD(clusters)
            
            calculation_info = {
                'Predicates points': data_points_array, # Predicate data + Constraint data

                'Level': clusters['Level'],
                'Cluster Id': clusters['Cluster Id'],
                'Parent level': clusters['Parent level'],
                'Parent cluster': clusters['Parent cluster'],

                'Data_Min': box['min'],     # For bound calculations
                'Data_Max': box['max'],     # For bound calculations
                'Count': count['count'],    # For bound calculations
                
                #f'Race1 Positive': cmplx_constraint_SPD['Race1_P'], # For constraint
                #f'Race1': cmplx_constraint_SPD['Race1'],            # For constraint
                #f'Race2 Positive': cmplx_constraint_SPD['Race2_P'], # For constraint
                #f'Race2': cmplx_constraint_SPD['Race2']             # For constraint
            }

            # Evaluate each aggregation for this cluster
            for idx, (agg_name, agg_expr) in enumerate(aggregations.items(), start=1):
                result = self.evaluate_aggregation(df_sliced, agg_expr)
                
                calculation_info[f'agg{idx}'] = result  # Add the aggregation result

            statistical_tree.append(calculation_info)


        # Create a DataFrame
        df_statistical_info = pd.DataFrame(statistical_tree)
        
        # Save to CSV
        df_statistical_info.to_csv('statistical_info.csv', index=False)

        return statistical_tree


    # Function to calculate the bounds for a given set of points
    def points_bounds(self, cluster_points):
        # Convert the list of lists into a NumPy array for easier manipulation
        np_data = np.array(cluster_points)

        min_vals = np.min(np_data, axis=0).tolist()
        max_vals = np.max(np_data, axis=0).tolist()

        return {'min': min_vals, 'max': max_vals}

    # Function to calculate the bounds for a given set of points
    def sum_bounds(self, box, num_points, df):
        sum_min=[]
        sum_max=[]
        sum_min.append(box['min'][0] * num_points)
        sum_max.append(box['max'][0] * num_points)
        sum_min.append(box['min'][1] * num_points)
        sum_max.append(box['max'][1] * num_points)
    
        return {'sum_'+df.columns[0]+'_min': sum_min[0], 'sum_'+df.columns[0]+'_max': sum_max[0], 'sum_'+df.columns[1]+'_min': sum_min[1], 'sum_'+df.columns[1]+'_max': sum_max[1]}


    def count(self, num_points):
        #min_count = 0
        count = num_points
        return {'count': count}

    def actual_sum(self, points):
        s = 0
        for point in points:
            s = s + point
        return {'sum': s}
    

    def avg_bounds(self, min_sum, max_sum, count):
        try:
            min_avg = min_sum/count
        except ZeroDivisionError:
            min_avg = "Nan" 

        try:
            max_avg = max_sum/count
        except ZeroDivisionError:
            max_avg = "Nan" 

        return {'min_avg': min_avg, 'max_avg': max_avg}

    '''
    def complex_constraint_Avg(self, avg1, avg2):
        try:
            min_cmplx_constraint = round(avg2['min_avg'] / avg1['max_avg'], 3)
        except ZeroDivisionError:
            min_cmplx_constraint = "Nan" 

        try:
            max_cmplx_constraint = round(avg2['max_avg'] / avg1['min_avg'], 3)
        except ZeroDivisionError:
            max_cmplx_constraint = "Nan" 

        return {'min_cmplx_constraint': min_cmplx_constraint, 'max_cmplx_constraint': max_cmplx_constraint}

    
    def complex_constraint_SPD(self, clusters):

        counter1_U=0
        for cluster in clusters['Data points']:
            if cluster[2] == 1 and cluster[3] >= 20000:
                counter1_U +=1

        counter2_U=0
        for cluster in clusters['Data points']:
            if cluster[2] == 2 and cluster[3] >= 20000:
                counter2_U +=1

        counter11_U=0
        for cluster in clusters['Data points']:
            if cluster[2] == 1:
                counter11_U +=1

        counter22_U=0
        for cluster in clusters['Data points']:
            if cluster[2] == 2:
                counter22_U +=1

        return {'Race1_P': counter1_U, 'Race1': counter11_U, 'Race2_P': counter2_U, 'Race2': counter22_U}    
    '''


        