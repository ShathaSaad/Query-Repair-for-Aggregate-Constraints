import pandas as pd
from SQL_operators import SQL_operators
from itertools import product, chain, combinations
import matplotlib.pyplot as plt
from Manhattan_distance import Manhattan_distance
import time
from itertools import product
from constraint_evaluation import constraint_evaluation

import os




class brute_force:

    def calculate_combinations(self, predicates):
        num_combinations = 1
        for predicate in predicates:
            num_combinations *= len(predicate['values'])
        return num_combinations

    def calculate(self, value1, value2):
        return abs(value1 - value2)
        
    def PossibleRef_allCombination(self, df_original, all_pred_possible_values, expression, aggregations, UserpredicateList, datasize, dataName):
        satisfied_conditions = []
        op = SQL_operators()
        counter = 0
        agg_counter = 0
        found = False
        possible_refinements = []
        check_counter = 0
        refinement_counter = 0

        # Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': pred_info['values']} for pred_info in all_pred_possible_values]
        
        # Generate all combinations of values for all predicates
        combinations = product(*[predicate['values'] for predicate in predicates])

        # Sort combinations by the first column (income) and then by the second column (children)
        #sorted_combinations = sorted(combinations, key=lambda x: (x[0], x[1]))
        
        # Construct possible refinements based on combinations
        for combination in combinations:
            refinement = {
            'refinements': [],
            'distance': 0  # Initialize distance
            }
            for pred_info, val in zip(predicates, combination):
                predicate_info = pred_info['predicate']
                refinement['refinements'].append ({
                    'column': predicate_info['column'],
                    'operator': predicate_info['operator'],
                    'value': val
                })


            # For demonstration, just add up differences between matched user predicates
            for pred in refinement['refinements']:
                for user_pred in UserpredicateList:
                    if pred['column'] == user_pred[0]:
                        pred_distance = self.calculate(pred['value'], user_pred[2])
                        refinement['distance'] += pred_distance
        
            possible_refinements.append(refinement)

        # Sort possible refinements by distance
        sorted_possible_refinements = sorted(possible_refinements, key=lambda x: x['distance'])
        
        # Iterate over all combinations
        start_time_comb = time.time()
        for refinement in sorted_possible_refinements:
            if found == True:
                break
            filtered_df = df_original.copy()
            similarity = refinement['distance']
            # Apply each predicate to the DataFrame
            for predicate in refinement['refinements']:
                counter += 1        
                operator = predicate['operator']
                value = predicate['value'] 
                column = predicate['column'] 

                if operator == '>=':            
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == '==':
                    filtered_df = filtered_df[filtered_df[column] == value]       
            
            # Evaluate the expression using the filtered DataFrame
            #satisfied, agg_counter = self.evaluate_expression(aggregations, expression, filtered_df, agg_counter)
            check_counter +=1
            satisfied, agg_counter, counter = self.custom_agg_func(op, filtered_df, counter, agg_counter, refinement['refinements'][0]['value'], refinement['refinements'][1]['value'] , similarity)
            if satisfied != []:
                refinement_counter += 1
                satisfied_conditions.append(satisfied)
                #if len(satisfied_conditions) == 7:
                    #break
            
            #filename = f'filtered_BruteForce_combination_{combination}_modified.csv'
            #filename = filename.replace(" ", "_").replace(",", "_")
            # Save to CSV
            #filtered_df.to_csv(filename, index=False)
            
        end_time_comb = time.time()  

        satisfied_conditions_df = pd.DataFrame(satisfied_conditions)
        satisfied_conditions_df.to_csv("satisfied_conditions_BruteForce.csv", index=False)
        elapsed_time = end_time_comb - start_time_comb

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
        
        elapsed_time_comb = end_time_comb - start_time_comb
        #print("Number of Combination:", self.calculate_combinations(predicates)) 
        print("Number of data access: ", counter)
        print("Number of checked", check_counter)
        print("Number of refinments", refinement_counter)
        print("Time taken Overall:", round(elapsed_time_comb, 3), "seconds") 
        #print("Number of Aggregation calculated: ", agg_counter)
        

    # Define a custom aggregation function
    def custom_agg_func(self, op, filtered_df, counter, agg_counter, pred1, pred2, similarity):
        satisfied = []

        data = pd.DataFrame(filtered_df)
        if data.empty:
            return satisfied, agg_counter, counter
        else:
            agg_counter+=1
            counter += 6
             
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
            '''
                     
            # For ACSIncome -------------------------------------------------------------------------
            property_num = 2 # -0.3 <= float(SPD) <= 0.2       
            male_positive_tuples_sex = op.filter("WhyN", filtered_df, "RAC1P", "==", 1.0) 
            male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, "PINCP", ">=", 50000.0) 
            female_positive_tuples_sex = op.filter("WhyN", filtered_df, "RAC1P", "==", 2.0)
            female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "PINCP", ">=", 50000.0) 
            male_count_tuples = op.filter("WhyN", filtered_df, "RAC1P", "==", 1.0) #count with filter condition
            female_count_tuples = op.filter("WhyN", filtered_df, "RAC1P", "==", 2.0) #count with filter condition

            male_positive = len(male_positive_tuples)
            female_positive = len(female_positive_tuples)
            male_count = len(male_count_tuples)
            female_count = len(female_count_tuples) 
            # ---------------------
            '''
            
            #For Healthcare -------------------------------------------------------------------------
            '''
            property_num = 1 # -0.1 <= float(SPD) <= 0.1
            male_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race1") 
            male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", True) 
 

            female_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race2")
            female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", True) 

            male_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race1") #count with filter condition
            female_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race2") #count with filter condition
            male_positive = len(male_positive_tuples)
            female_positive = len(female_positive_tuples)
            male_count = len(male_count_tuples)
            female_count = len(female_count_tuples)
            
            property_num = 1
            
            male_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", 1) 
            male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", 1) 
 

            female_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", 2)
            female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", 1) 

            male_count_tuples = op.filter("WhyN", filtered_df, "race", "==", 1) #count with filter condition
            female_count_tuples = op.filter("WhyN", filtered_df, "race", "==", 2) #count with filter condition
            
            male_positive = len(male_positive_tuples)
            female_positive = len(female_positive_tuples)
            male_count = len(male_count_tuples)
            female_count = len(female_count_tuples)
            # ---------------------
            
            
            property_num = 2 #-0.3 <= float(SPD) <= 0.2
            male_positive_tuples_sex = op.filter("WhyN", filtered_df, "age-group", "==", "group1") 
            male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", True) 
 

            female_positive_tuples_sex = op.filter("WhyN", filtered_df, "age-group", "==", "group2")
            female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", True) 

            male_count_tuples = op.filter("WhyN", filtered_df, "age-group", "==", "group1") #count with filter condition
            female_count_tuples = op.filter("WhyN", filtered_df, "age-group", "==", "group2") #count with filter condition
            
            male_positive = len(male_positive_tuples)
            female_positive = len(female_positive_tuples)
            male_count = len(male_count_tuples)
            female_count = len(female_count_tuples)   
            
            # End Health ---------------------------------------------------------------------------
            '''
            # For TPC-H ----------------------------------------------------------------------------
            '''
            property_num = 1
            sum_supplier = sum(filtered_df['s_acctbal']) 
            count_supplier = len(filtered_df['s_suppkey']) 
            try:
                constraint = round(sum_supplier/count_supplier, 5)
                if 2000 <= constraint <= 5000:
                    possibleRefinments.append({'combination': combination, 'constraint': constraint})
            except ZeroDivisionError:
                pass
            
            property_num = 2
            sum_supplier = sum(filtered_df['s_acctbal']) 
            count_supplier = len(filtered_df['s_suppkey']) 
            try:
                constraint = round(sum_supplier/count_supplier, 5)
                if constraint <= 7000:
                    possibleRefinments.append({'combination': combination, 'constraint': constraint})
            except ZeroDivisionError:
                pass
            # ---------------------------------------------------------------------------------------
            '''

            
            
            if (male_count == 0 and female_count != 0):
                SPD = round(0 - (female_positive / female_count), 4) #arithmatic expression
            elif (female_count ==0 and male_count != 0):
                SPD = round((male_positive / male_count) - 0, 4) #arithmatic expression
            elif (female_count ==0 and male_count == 0):
                SPD = 0
            else:
            
            #try:
                SPD = round((male_positive / male_count) - (female_positive / female_count), 4) #arithmatic expression
        
            if 0.0 <= SPD <= 0.1: 
                satisfied = ({'combination': (pred1, pred2), 'SPD': SPD, "Similarity": similarity})
            #except ZeroDivisionError:
                #pass
        return satisfied, agg_counter, counter


    def evaluate_aggregation(self, df, formula):
        """
        Evaluates the aggregation formula by counting rows that match the condition.
        This method now uses pandas boolean indexing to evaluate conditions.
        """
        # Remove 'count(' and ')' from the formula to get the actual condition
        condition = formula.replace('count(', '').replace(')', '')

        try:
            # Convert condition into executable boolean indexing expressions
            if 'and' in condition:
                # Split by 'and' to handle multiple conditions
                conditions = condition.split('and')
                conditions = [cond.strip() for cond in conditions]

                # Create a boolean mask by evaluating each condition separately
                mask = pd.Series([True] * len(df))  # Start with all True mask
                for cond in conditions:
                    mask &= df.eval(cond)  # Evaluate each condition on the DataFrame
            else:
                # Single condition
                mask = df.eval(condition)
            
            # Return the count of rows that match the condition
            return mask.sum()  # Count the True values in the boolean mask
        except Exception as e:
            raise ValueError(f"Error evaluating aggregation formula '{formula}': {e}")  
    
    def evaluate_aggregation(self, df, formula):
            """
                Evaluates the aggregation formula by counting rows that match the condition.
            """
            # Remove 'count(' and ')' from the formula to extract the condition
            condition = formula.replace('count(', '').replace(')', '').strip()

        #try:
            # Split the condition into individual conditions that can be evaluated
            # Handle condition like 'PINCP >= 20000 and SEX == 1'
            if ' and ' in condition:
                conditions = condition.split(' and ')
                query_result = df
                for cond in conditions:
                    column, operator, value = self.parse_condition(cond)
                    query_result = query_result[self.apply_condition(query_result, column, operator, value)]
            else:
                column, operator, value = self.parse_condition(condition)
                query_result = df[self.apply_condition(df, column, operator, value)]

            # Return the number of rows that match the condition
            return query_result

    def PossibleRef_incrementalAgg1(self, df_original, all_pred_possible_values, UserpredicateList, dataName, dataSize, query_num):

        possibleRefinments = []
        pre_agg = []
        op = SQL_operators()
        elapsed_time_agg = 0
        elapsed_time_post_agg = 0

        # Sorting & Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': sorted(pred_info['values']) if pred_info['predicate']['type'] == 'numerical' else pred_info['values']} for pred_info in all_pred_possible_values]
        print("--------------------------")
        print(predicates)
        print("--------------------------")
        # Generate all combinations of values for all predicates
        combinations = product(*[predicate['values'] for predicate in predicates])

        start_time_comb = time.time()   
        for combination in combinations:
                filtered_df = filtered_df = df_original.copy()

                # Pre-aggregation
                for i, predicate in enumerate(predicates):
                    value = combination[i]
                    column = predicate['predicate']['column']

                    #result = filtered_df.groupby(['income', 'num-children']).agg(lambda x: self.custom_agg_func(op, filtered_df))

                    filtered_df = filtered_df[filtered_df[column] == value]


                start_time_agg = time.time()
                '''
                male_positive = op.filter("WhyN", filtered_df, "Gender", "==", "M")
                male_positive = len(op.filter("WhyN", male_positive, "Hours", "<=", 60))

                female_positive = op.filter("WhyN", filtered_df, "Gender", "==", "F")
                female_positive = len(op.filter("WhyN", female_positive, "Hours", "<=", 60)) 

                male_count = len(op.filter("WhyN", filtered_df, "Gender", "==", "M")) #count with filter condition
                female_count = len(op.filter("WhyN", filtered_df, "Gender", "==", "M")) #count with filter condition

                '''

                # Pre-aggregate
                male_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race1") 
                male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", True) 
 

                female_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race2")
                female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", True) 
                
                male_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race1") #count with filter condition
                female_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race2") #count with filter condition
                    
                male_positive = len(male_positive_tuples)
                female_positive = len(female_positive_tuples)
                male_count = len(male_count_tuples)
                female_count = len(female_count_tuples)
                
                end_time_agg = time.time()
                elapsed_time_agg += end_time_agg - start_time_agg
                
                pre_agg.append({
                'combination': tuple(combination), 
                'male_positive': male_positive, 
                'female_positive': female_positive, 
                'male_count': male_count, 
                'female_count': female_count}) 
                '''
                'male_positive_tuples': male_positive_tuples['id'],
                'female_positive_tuples': female_positive_tuples['id'], 
                'male_count_tuples': male_count_tuples['id'], 
                'female_count_tuples':female_count_tuples['id']})
                '''
        
        combinations = product(*[predicate['values'] for predicate in predicates])
        operators = []


        for predicate in UserpredicateList:
            operators.append(predicate[1])
        
        for predicate in UserpredicateList:
            operators.append(predicate[1])
        
        for combination in combinations:   
                maleP = 0
                femaleP = 0
                male = 0
                female = 0    
                value1 = combination[0]
                value2 = combination[1]
                value3 = combination[2]

                start_time_post_agg = time.time()

                for item in pre_agg:
                    if operators[0] == '>=':
                        if item['combination'][0] >= value1:
                            filter1 = item['combination'][0] 
                    elif operators[0] == '<=':
                        if item['combination'][0] <= value1:
                            filter1 = item['combination'][0] 
                    elif operators[0] == '==':
                        if item['combination'][0] == value1:
                            filter1 = item['combination'][0] 

                    if operators[1] == '>=':
                        if item['combination'][1] >= value2:
                            filter2 = item['combination'][1]
                    elif operators[1] == '<=':
                        if item['combination'][1] <= value2:
                            filter2 = item['combination'][1]
                    elif operators[1] == '==':
                        if item['combination'][1] == value2:
                            filter2 = item['combination'][1]

                    if operators[2] == '>=':
                        if item['combination'][2] >= value3:
                            filter3 = item['combination'][2]
                    elif operators[2] == '<=':
                        if item['combination'][2] <= value3:
                            filter3 = item['combination'][2]
                    elif operators[2] == '==':
                        if item['combination'][2] == value3:
                           filter3 = item['combination'][2]
                        else:
                            filter3 = "nan"

                    if item['combination'] == (filter1, filter2, filter3):
                        maleP += item['male_positive']
                        femaleP += item['female_positive']
                        male += item['male_count']
                        female += item['female_count']

                try:
                    SPD = round((maleP / male) - (femaleP / female), 2) #arithmatic expression
                    #print("Combination:", combination, "Male Positive:", maleP, "Male Count:", male, "Female Positive:", femaleP, "Female Count:", female, "SPD:", SPD)
                    end_time_post_agg = time.time()
                    elapsed_time_post_agg += end_time_post_agg - start_time_post_agg
                    if -0.01 <= float(SPD) <= 0.01:
                        possibleRefinments.append({'combination': combination, 'SPD': SPD})
                except ZeroDivisionError:
                    pass
        end_time_comb = time.time() 

        #print("***** possiblePredicates: ")
        #for item in possibleRefinments:
            #print(item)
        #print("Time taken:", round(elapsed_time_agg + start_time_post_agg, 3), "seconds")
        print("Time taken Overall:", round(end_time_comb - start_time_comb, 3), "seconds") 
        print("Number of Combination:", self.calculate_combinations(predicates)) 
        print("Number of Predicates:", len(predicates))  
        print("Dataset Name:", dataName) 
        print("Dataset Size:", dataSize)   


    def PossibleRef_incrementalAgg(self, df_original, all_pred_possible_values, UserpredicateList, dataName, dataSize, query_num):
        possibleRefinments = []
        basicAgg = []
        op = SQL_operators()
        elapsed_time_agg = 0
        elapsed_time_intersection = 0
        

        # Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': pred_info['values']} for pred_info in all_pred_possible_values]
        print(predicates)

        start_time_comb = time.time() 
        # Iterate over all combinations
        for predicate in all_pred_possible_values:
            operator = predicate['predicate']['operator']
            column = predicate['predicate']['column']
            for value in predicate['values']:
                filtered_df = df_original.copy()

                if operator == '>=':
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == '==':
                    filtered_df = filtered_df[filtered_df[column] == value]

                start_time_agg = time.time()
                '''
                male_positive_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 1.0, "credit", "== yes")  
                female_positive_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 2.0, "credit", "== yes")
                male_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 1.0) #count with filter condition
                female_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 2.0) #count with filter condition
                '''
                
                #For Adult
                '''
                male_positive_tuples_sex = op.filter("WhyN", filtered_df, "SEX", "==", 1.0) 
                male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, "WKHP", ">", 50000.0) 
                female_positive_tuples_sex = op.filter("WhyN", filtered_df, "SEX", "==", 2.0)
                female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "WKHP", ">", 50000.0) 
                male_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 1.0) #count with filter condition
                female_count_tuples = op.filter("WhyN", filtered_df, "SEX", "==", 2.0) #count with filter condition
                
                '''
                #For Healthcare
                male_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race1") 
                male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", True) 
 

                female_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", "race2")
                female_positive_tuples = op.filter("WhyN", filtered_df, "label", "==", True) 
                
                male_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race1") #count with filter condition
                female_count_tuples = op.filter("WhyN", filtered_df, "race", "==", "race2") #count with filter condition
                
                male_positive = len(male_positive_tuples)
                female_positive = len(female_positive_tuples)
                male_count = len(male_count_tuples)
                female_count = len(female_count_tuples)
                
                basicAgg.append({'predicate': predicate['predicate'], 'value': value, 'male_positive': male_positive, 
                'female_positive': female_positive, 'male_count': male_count, 'female_count': female_count, 'male_positive_tuples': male_positive_tuples['id'],
                'female_positive_tuples': female_positive_tuples['id'], 'male_count_tuples': male_count_tuples['id'], 'female_count_tuples':female_count_tuples['id']})
        
                
                end_time_agg = time.time()

                elapsed_time_agg += end_time_agg - start_time_agg


                
        
        # Generate all combinations of values for all predicates
        combinations = product(*[predicate['values'] for predicate in predicates])

        for combination in combinations:
            #print("For combination: ", combination)
            maleP = []
            femaleP = []
            male = []
            female = []

            # Apply each predicate to the DataFrame
            for i, predicate in enumerate(predicates):
                value = combination[i]
                operator = predicate['predicate']['operator']
                column = predicate['predicate']['column']


                for item in basicAgg:
                    if value == item['value']:
                        maleP.append(item['male_positive_tuples'])
                        femaleP.append(item['female_positive_tuples'])
                        male.append(item['male_count_tuples'])
                        female.append(item['female_count_tuples'])


            # Find the intersection
            start_time_intersection = time.time()
            intersection1 = maleP[0].index.intersection(maleP[1].index)
            intersection2 = femaleP[0].index.intersection(femaleP[1].index)
            intersection3 = male[0].index.intersection(male[1].index)
            intersection4 = female[0].index.intersection(female[1].index)

            # Get the matching tuples
            match_maleP= len(maleP[0].loc[intersection1])
            match_femaleP= len(femaleP[0].loc[intersection2])
            match_male= len(male[0].loc[intersection3])
            match_female= len(female[0].loc[intersection4])
            
      
            try:
                SPD = round((match_maleP / match_male) - (match_femaleP / match_female), 2) #arithmatic expression
                end_time_intersection = time.time()
                elapsed_time_intersection += end_time_intersection - start_time_intersection
                #print("Combination:", combination, "Male Positive:", match_maleP, "Male Count:", match_male, "Female Positive:", match_femaleP, "Female Count:", match_female, "SPD:", SPD)

                if -0.5 <= float(SPD) <= 0.5:
                    possibleRefinments.append({'combination': combination, 'predicates': [predicate['predicate'] for predicate in predicates], 'SPD': SPD})
                    possibleRefinments.append({'combination': combination, 'SPD': SPD})
            except ZeroDivisionError:
                pass

        end_time_comb = time.time() 
        #print("***** possiblePredicates: ")
        #for item in possibleRefinments:
            #print(item)
        elapsed_time_comb = end_time_comb - start_time_comb
        elapsed_time_Agg_Inters = elapsed_time_intersection + elapsed_time_agg
        print("Time taken:", round(elapsed_time_Agg_Inters,3) , "seconds") 
        #print("Time taken Overall:", round(elapsed_time_comb,3) , "seconds") 
        print("Number of Combination:", self.calculate_combinations(predicates)) 
        print("Number of Predicates:", len(predicates))   
        print("Dataset Name:", dataName) 
        print("Dataset Size:", dataSize)    
        self.output_to_csv(dataName, "Incremental Agg", dataSize, round(elapsed_time_Agg_Inters,3), self.calculate_combinations(predicates), len(predicates), query_num)


    def output_to_csv(self, data_name, type, dataSize, time_taken, combination_num, predicates_num, numRefinment, query_num, property_num):
        # Generate the file name based on the dataset name
        file_name = f'{data_name.lower().replace(" ", "_")}_runtime_data.csv'
        #file_name = f'{data_name.lower().replace(" ", "_")}_runtime_data.csv'
        directory = '/Users/Shatha/Downloads/inputData/BruteForce_Results'

        full_file_path = os.path.join(directory, file_name)

        # Check if the file already exists
        if os.path.exists(full_file_path):
        # Read the existing CSV file into a DataFrame
            df_existing = pd.read_csv(full_file_path)

            # Add the new data to the existing DataFrame
            df_new = pd.DataFrame({
                'Data Name': [data_name],
                'Data Size': [dataSize],
                'type': [type],
                'Time Taken': [time_taken],
                'Combination Number': [combination_num],
                'Predicates Number': [predicates_num],
                'Possible Refinment Numbers': [numRefinment],
                'Query Number': [query_num],
                'property_num': [property_num]
            })

            # Concatenate the new data with the existing DataFrame
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)

            # Save the updated DataFrame to the CSV file
            df_updated.to_csv(full_file_path, index=False)

            # Display the updated DataFrame
            print(df_updated)

        else:
            # If the file does not exist, create a new DataFrame with the new data and save it to the CSV file
            df_new = pd.DataFrame({
                'Data Name': [data_name],
                'Data Size': [dataSize],
                'type': [type],
                'Time Taken': [time_taken],
                'Combination Number': [combination_num],
                'Predicates Number': [predicates_num],
                'Possible Refinment Numbers': [numRefinment],
                'Query Number': [query_num],
                'property_num': [property_num]
            })

            # Concatenate the directory and file name to create the full file path
            df_new.to_csv(full_file_path, index=False)

            # Display the new DataFrame
            print(df_new)

    def PossibleRef_allCombination1(self, df_original, all_pred_possible_values, UserpredicateList, dataName, dataSize, query_num):
        possibleRefinments = []
        op = SQL_operators()
        elapsed_time_agg = 0

        # Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': pred_info['values']} for pred_info in all_pred_possible_values]

        # Generate all combinations of values for all predicates
        combinations = product(*[predicate['values'] for predicate in predicates])

        # Iterate over all combinations
        start_time_comb = time.time() 
        filtered_df = df_original.copy()

        # Apply each predicate to the DataFrame
        for i, predicate in enumerate(predicates):
            
            if i == 0:
                column = predicate['predicate'][0]
                filtered_df = filtered_df[filtered_df[column] >= 100]
            if i == 1:
                column = predicate['predicate'][0]
                filtered_df = filtered_df[filtered_df[column] >= 0]
        
        start_time = time.time()

            
            #For Healthcare -------------------------------------------------------------------------
            
        property_num = 1 # -0.1 <= float(SPD) <= 0.1
        male_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", 1) 
        male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", 1) 
 

        female_positive_tuples_sex = op.filter("WhyN", filtered_df, "race", "==", 2)
        female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", 1) 

        male_count_tuples = op.filter("WhyN", filtered_df, "race", "==", 1) #count with filter condition
        female_count_tuples = op.filter("WhyN", filtered_df, "race", "==", 2) #count with filter condition
            
        male_positive = len(male_positive_tuples)
        female_positive = len(female_positive_tuples)
        male_count = len(male_count_tuples)
        female_count = len(female_count_tuples)
        #print(male_positive)
        #print(male_count)
        #print(female_positive)
        #print(female_count)
        # ---------------------
        '''
            property_num = 2 #-0.3 <= float(SPD) <= 0.2
            male_positive_tuples_sex = op.filter("WhyN", filtered_df, "age-group", "==", "group1") 
            male_positive_tuples = op.filter("WhyN", male_positive_tuples_sex, 'label', "==", True) 
 

            female_positive_tuples_sex = op.filter("WhyN", filtered_df, "age-group", "==", "group2")
            female_positive_tuples = op.filter("WhyN", female_positive_tuples_sex, "label", "==", True) 

            male_count_tuples = op.filter("WhyN", filtered_df, "age-group", "==", "group1") #count with filter condition
            female_count_tuples = op.filter("WhyN", filtered_df, "age-group", "==", "group2") #count with filter condition
            
            male_positive = len(male_positive_tuples)
            female_positive = len(female_positive_tuples)
            male_count = len(male_count_tuples)
            female_count = len(female_count_tuples)   
            
            # End Health ---------------------------------------------------------------------------
        '''

        try:
                SPD = round((male_positive / male_count) - (female_positive / female_count), 5) #arithmatic expression
                end_time_agg = time.time()
                #print("Combination:", combination, "Male Positive:", male_positive, "Male Count:", male_count, "Female Positive:", female_positive, "Female Count:", female_count, "SPD:", SPD)
                #elapsed_time_agg += end_time_agg - start_time_agg
                #print("Time taken:", elapsed_time_agg, "seconds")
                print("SPD for brute force = ", SPD)
                if 0.1 <= float(SPD) <= 0.2:
                    possibleRefinments.append({'predicates': [predicate['predicate'] for predicate in predicates], 'SPD': SPD})
        except ZeroDivisionError:
                print('ZeroDivisionError')
    
        

        end_time = time.time()  
        numRefinment = 0
        print("***** possiblePredicates: ")
        for item in possibleRefinments:
            print(item)
            numRefinment+=1
        
        #print("Time taken:", round(elapsed_time_agg, 3), "seconds")
        elapsed_time = end_time - start_time
        print("Time taken Overall for all combination:", round(elapsed_time, 3), "seconds") 

        
        
        
        
        




    
                 