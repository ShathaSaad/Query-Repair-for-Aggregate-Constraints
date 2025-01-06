from unittest import skip
import pandas as pd
import re
import numpy as np
from ExpressionNode import ExpressionNode



class constraint_evaluation:
    # Method to extract boundary values from the expression
    def extract_boundary_values(self, expression):

        # Regex pattern to match floating-point numbers or integers in the expression
        pattern = r'[-+]?\d*\.\d+|\d+'
        matches = re.findall(pattern, expression)

        if len(matches) < 2:
            raise ValueError("Expression must contain at least two boundary values (e.g., '0.0 <= expression <= 0.2')")
        
        # Convert matches to floats and return them
        lower_bound = float(matches[0])
        upper_bound = float(matches[-1])
        return lower_bound, upper_bound

    # Function to extract and evaluate the core part of the expression (e.g., agg1 / agg2 - agg3 / agg4)
    def extract_core_expression(self, expression):
        # Extract the core part of the expression (inside parentheses or arithmetic expressions)
        match = re.search(r'<=\s*(.*)\s*<=', expression)
        if match:
            return match.group(1)
        else:
            # If there's no comparison in the expression, return the full expression
            return expression
    def apply_arithmetic_operation(self, operand1, operand2, operator):
        """Performs basic arithmetic operations safely."""
        if operator == "+":
            return operand1 + operand2
        elif operator == "-":
            return operand1 - operand2
        elif operator == "*":
            return operand1 * operand2
        elif operator == "/":
            return self.safe_div(operand1, operand2)
        else:
            raise ValueError(f"Unknown operator {operator}")

    def sum_values(self, values):
        """Helper function to sum values, handling possible multi-dimensional arrays."""
        if isinstance(values[0], (np.ndarray, list)):
            # If the values are arrays or lists, flatten them first and sum the result
            return sum([np.sum(val) for val in values])
        else:
            # Otherwise, sum scalar values normally
            return np.sum(values)

    def safe_div(self, x, y):
        """A helper function for safe division to avoid ZeroDivisionError."""
        if y != 0:
            result =  x / y 
        else: 
            result = 0
        return result  

    def parse_and_evaluate_expression(self, filtered_df, expression):
        """Evaluates the expression with proper handling of parentheses."""
        
        def evaluate(tokens):
            """Helper function to evaluate tokens list without parentheses."""
            stack = []
            current_value = None

            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token in ['+', '-', '*', '/']:
                    operator_token = token
                else:
                    if token in filtered_df.columns:
                        col_values = filtered_df[token].values
                        token_value = self.sum_values(col_values)
                    else:
                        token_value = float(token)

                    if current_value is None:
                        current_value = token_value
                    else:
                        current_value = self.apply_arithmetic_operation(current_value, token_value, operator_token)
                
                idx += 1
            
            return current_value

        # Tokenizing the expression and handling parentheses
        def tokenize_expression(expr):
            """Tokenizes the arithmetic expression, handling parentheses."""
            tokens = re.findall(r'[\w\.]+|[\+\-\*/\(\)]', expr)
            return tokens
        
        tokens = tokenize_expression(expression)
        stack = []

        # Process the tokens with a stack to evaluate expressions inside parentheses first
        output = []
        for token in tokens:
            if token == "(":
                stack.append(output)
                output = []
            elif token == ")":
                result = evaluate(output)
                output = stack.pop()
                output.append(str(result))
            else:
                output.append(token)
        
        # Finally evaluate the expression outside parentheses
        return evaluate(output)
    
    def evaluate_constraint1(self, filtered_df, expression, conditions, agg_counter, similarity, type, constraint_type, concrete_values=None):
        
        satisfied = []
        Not_satisfied = False
        result = None
        np.seterr(invalid='ignore')
        satisfies_all = False

        data = pd.DataFrame(filtered_df)
                    
        if data.empty:
            Not_satisfied = True
            return satisfied, agg_counter, Not_satisfied, result
        else:
            try:
                # Extract boundary values from the expression
                lower_bound, upper_bound = self.extract_boundary_values(expression)

                # Evaluate the exact result of the expression (SPD or another calculation)
                core_expression = self.extract_core_expression(expression)
                result = round(self.parse_and_evaluate_expression(data, core_expression), 4)
                #print(conditions, result)
                # Check if the result satisfies the boundary conditions for all conditions
                if constraint_type == 1:
                    satisfies_all = lower_bound <= result <= upper_bound
                if constraint_type == 2:
                    satisfies_all = result <= upper_bound

                # Construct the satisfaction result based on the type and conditions
                agg_counter += 1
                if satisfies_all:
                    if type == "ranges":
                        satisfied = {
                            "conditions": conditions,
                            "Concrete Values": concrete_values,
                            "Result": result,
                            "Similarity": similarity,
                            "Range Satisfaction": "Full"
                        }
                    else:
                        satisfied = {
                            "conditions": conditions,
                            "Result": result,
                            "Similarity": similarity,
                            "Range Satisfaction": "Full"
                        }
                else:
                    Not_satisfied = True    

            except ZeroDivisionError:
                pass

            return satisfied, agg_counter, Not_satisfied, result


    def evaluate_constraint(self, filtered_df, expression, condition1, condition2, condition3, agg_counter, similarity, type, concerete_value1=0, 
    concerete_value2=0, concerete_value3=0):
        satisfied = []
        result = None
        np.seterr(invalid='ignore')

        data = pd.DataFrame(filtered_df)
                
        if data.empty:
            return satisfied, agg_counter
        else:
            try:
                #agg_counter += 1
                # Extract boundary values from the expression
                lower_bound, upper_bound = self.extract_boundary_values(expression)

                # Evaluate the exact result of the expression (SPD or another calculation)
                # Extract the core expression and evaluate it explicitly
                core_expression = self.extract_core_expression(expression)
                result = round(self.parse_and_evaluate_expression(data, core_expression), 4)
                #print("condition1", condition1, "condition2", condition2, "Result:", result) 
            
                # Check if the result satisfies the boundary conditions
                if type == "ranges":
                    if lower_bound <= result <= upper_bound:
                        satisfied = {
                            "condition1": condition1, "Concrete Vlaues1": concerete_value1,
                            "condition2": condition2, "Concrete Vlaues2": concerete_value2,
                            "condition3": condition3, "Concrete Vlaues3": concerete_value3,
                            "Result": result,
                            "Similarity": similarity,
                            "Range Satisfaction": "Full"
                        }
                else:
                    if lower_bound <= result <= upper_bound:
                        satisfied = {
                            "condition1": condition1,
                            "condition2": condition2, 
                            "condition2": condition3, 
                            "Result": result,
                            "Similarity": similarity,
                            "Range Satisfaction": "Full"
                        }

            except ZeroDivisionError:
                pass
            #print("condition1", condition1, "condition2", condition2, "Result:", result) 
            
            return satisfied, agg_counter

    def cardinality(self, filtered_df, counter, agg_counter, condition1, condition2):
        satisfied = []

        data = pd.DataFrame(filtered_df)
        if data.empty:
            return satisfied, agg_counter
        else:
            count_race1 = data['agg1'].sum()


            if 1 <= count_race1 and count_race1 <= 4:
                satisfied = {
                    "condition1": condition1, 
                    "condition2": condition2, 
                    "Result": [round(count_race1,4), round(count_race1,4)],  # Store as a list
                    "Range Satisfaction": "Full"
                }  
            elif (1 <= count_race1 <= 4 and 4 < count_race1) or (count_race1 <= 1 and 1 < count_race1 <= 4) or (
                count_race1 <= 1 and  count_race1 >= 4):
                satisfied = {
                    "condition1": condition1, 
                    "condition2": condition2, 
                    "Result": [round(count_race1,4), round(count_race1,4)],  # Store as a list
                    "Range Satisfaction": "Partial"
                } 
        return satisfied, agg_counter

    def calculate_spd_fully(self, filtered_df, counter, agg_counter, condition1, condition2): #, similarity):
        satisfied = []

        if filtered_df.empty:
            return satisfied, agg_counter
        else:
            agg_counter = agg_counter + 1
            count_race1_positive = filtered_df['Race1 Positive'].sum()
            count_race1 = filtered_df['Race1'].sum()
            count_race2_positive = filtered_df['Race2 Positive'].sum()
            count_race2 = filtered_df['Race2'].sum()

            try:
                # Calculate SPD
                spd = round((count_race1_positive / count_race1) - (count_race2_positive / count_race2), 4)
                #print("SPD for Fully filtered clusters = ", spd)
                #if 0.0 <= spd <= 0.2:
                satisfied = ({"condition1": condition1,
                    "condition2": condition2,
                          "SPD": spd, 
                          "counter": counter})
                          #"similarity": similarity})
            except ZeroDivisionError:
                pass
            return satisfied, agg_counter

    def evaluate_constraint2(self, filtered_df, expression, condition1, condition2, agg_counter, similarity, type, concerete_value1=0, 
        concerete_value2=0, concerete_value3=0):
            satisfied = []
            result = None
            np.seterr(invalid='ignore')

            data = pd.DataFrame(filtered_df)
                    
            if data.empty:
                return satisfied, agg_counter
            else:
                #try:
                    #agg_counter += 1
                    # Extract boundary values from the expression
                    lower_bound, upper_bound = self.extract_boundary_values(expression)

                    # Evaluate the exact result of the expression (SPD or another calculation)
                    # Extract the core expression and evaluate it explicitly
                    core_expression = self.extract_core_expression(expression)
                    result = round(self.parse_and_evaluate_expression(data, core_expression), 4)
                    #print("condition1", condition1, "condition2", condition2, "Result:", result) 
                
                    # Check if the result satisfies the boundary conditions
                    if type == "ranges":
                        if lower_bound <= result <= upper_bound:
                            satisfied = {
                                "condition1": condition1, "Concrete Vlaues1": concerete_value1,
                                "condition2": condition2, "Concrete Vlaues2": concerete_value2,
                                "Result": result,
                                "Similarity": similarity,
                                "Range Satisfaction": "Full"
                            }
                    else:
                        if lower_bound <= result <= upper_bound:
                            satisfied = {
                                "condition1": condition1,
                                "condition2": condition2, 
                                "Result": result,
                                "Similarity": similarity,
                                "Range Satisfaction": "Full"
                            }

                #except ZeroDivisionError:
                    #pass
                #print("condition1", condition1, "condition2", condition2, "Result:", result) 
                
                    return satisfied, agg_counter

    def cardinality(self, filtered_df, counter, agg_counter, condition1, condition2):
        satisfied = []

        data = pd.DataFrame(filtered_df)
        if data.empty:
            return satisfied, agg_counter
        else:
            count_race1 = data['agg1'].sum()


            if 1 <= count_race1 and count_race1 <= 4:
                satisfied = {
                    "condition1": condition1, 
                    "condition2": condition2, 
                    "Result": [round(count_race1,4), round(count_race1,4)],  # Store as a list
                    "Range Satisfaction": "Full"
                }  
            elif (1 <= count_race1 <= 4 and 4 < count_race1) or (count_race1 <= 1 and 1 < count_race1 <= 4) or (
                count_race1 <= 1 and  count_race1 >= 4):
                satisfied = {
                    "condition1": condition1, 
                    "condition2": condition2, 
                    "Result": [round(count_race1,4), round(count_race1,4)],  # Store as a list
                    "Range Satisfaction": "Partial"
                } 
        return satisfied, agg_counter

    def calculate_spd_fully(self, filtered_df, counter, agg_counter, condition1, condition2): #, similarity):
        satisfied = []

        if filtered_df.empty:
            return satisfied, agg_counter
        else:
            agg_counter = agg_counter + 1
            count_race1_positive = filtered_df['Race1 Positive'].sum()
            count_race1 = filtered_df['Race1'].sum()
            count_race2_positive = filtered_df['Race2 Positive'].sum()
            count_race2 = filtered_df['Race2'].sum()

            try:
                # Calculate SPD
                spd = round((count_race1_positive / count_race1) - (count_race2_positive / count_race2), 4)
                #print("SPD for Fully filtered clusters = ", spd)
                #if 0.0 <= spd <= 0.2:
                satisfied = ({"condition1": condition1,
                    "condition2": condition2,
                          "SPD": spd, 
                          "counter": counter})
                          #"similarity": similarity})
            except ZeroDivisionError:
                pass
            return satisfied, agg_counter
           

    def calculate_spd_partially(self, filtered_df, condition1, condition2, agg_counter):
        satisfied = []  

        data = pd.DataFrame(filtered_df)

        if data.empty:
            return satisfied, agg_counter
        else:
            agg_counter = agg_counter + 1
            # Initialize sums for lower and upper bounds
            count_race1_positive_lower = 0
            count_race1_lower = 0
            count_race2_positive_lower = 0
            count_race2_lower = 0
    
            count_race1_positive_upper = 0
            count_race1_upper = 0
            count_race2_positive_upper = 0
            count_race2_upper = 0
    
            # Iterate over each row in the DataFrame
            for _, row in data.iterrows():
                if row['Satisfy'] == 'Full':
                    # Use the values as both lower and upper bounds
                    count_race1_positive_lower += row['Race1 Positive']
                    count_race1_lower += row['Race1']
                    count_race2_positive_lower += row['Race2 Positive']
                    count_race2_lower += row['Race2']

                    count_race1_positive_upper += row['Race1 Positive']
                    count_race1_upper += row['Race1']
                    count_race2_positive_upper += row['Race2 Positive']
                    count_race2_upper += row['Race2']
        
                elif row['Satisfy'] == 'Partial':
                    # Use values as upper bounds, and 0 as lower bounds
                    count_race1_positive_upper += row['Race1 Positive']
                    count_race1_upper += row['Race1']
                    count_race2_positive_upper += row['Race2 Positive']
                    count_race2_upper += row['Race2']

            try: 
                # Calculate lower bound
                E1_Min = round((count_race1_positive_lower / count_race1_upper), 4)       
            except ZeroDivisionError:
                E1_Min = "undefined"
                pass
            try: 
                # Calculate lower bound
                E2_Min = round((count_race2_positive_lower / count_race2_upper), 4)       
            except ZeroDivisionError:
                E2_Min = "undefined"
                pass

            try: 
                # Calculate upper bound
                E1_Max = round((count_race1_positive_upper / count_race1_lower), 4)      
            except ZeroDivisionError:
                E1_Max = "undefined"
                pass
            try: 
                # Calculate upper bound
                E2_Max = round((count_race2_positive_upper / count_race2_lower), 4)      
            except ZeroDivisionError:
                E2_Max = "undefined"
                pass
            
            if E1_Min != "undefined" and E2_Max != "undefined":
                spd_lower = round(E1_Min - E2_Max, 4)
            else:
               spd_lower = "undefined" 
            if E1_Max != "undefined" and E2_Min != "undefined":
                spd_upper = round(E1_Max - E2_Min, 4)
            else:
                spd_upper= "undefined"

            #print("condition1", condition1, "condition2", condition2, "SPD for Partial filtered clustered = [", spd_lower, spd_upper, "]")
            if spd_lower != "undefined" and spd_upper != "undefined":
                if 0.0 <= spd_lower and spd_upper <= 0.1:
                    satisfied = {
                        "condition1": condition1,
                        "condition2": condition2,
                        "SPD": [spd_lower, spd_upper] # Store as a list
                    }  

            return satisfied, agg_counter
     

    def calculate_spd_partially_ranges(self, filtered_clusters_list_df, income_range, num_children_range, agg_counter):
        satisfied = []      

        # Initialize sums for lower and upper bounds
        count_race1_positive_lower_bound = 0
        count_race1_lower_bound = 0
        count_race2_positive_lower_bound = 0
        count_race2_lower_bound = 0

        count_race1_positive_upper_bound = 0
        count_race1_upper_bound = 0
        count_race2_positive_upper_bound = 0
        count_race2_upper_bound = 0
        agg_counter = agg_counter + 1

        # Iterate over each row in the DataFrame to calculate lower bound
        #Actual count for Fully and Zero for Partial
        for _, row in filtered_clusters_list_df.iterrows():
            #Upper Bound, 
            #Actual count for fully and partial for upper bound
            #if row['Bound'] == "Both":
            count_race1_positive_upper_bound += row['Race1 Positive']
            count_race1_upper_bound += row['Race1']
            count_race2_positive_upper_bound += row['Race2 Positive']
            count_race2_upper_bound += row['Race2']

            #Lower Bound,
            #Actual count for fully and Zero for partial for upper bound
            #if row['Bound'] == "Lower" or row['Bound'] == "Both":
            if row['Satisfy'] == 'Full':
                count_race1_positive_lower_bound += row['Race1 Positive']
                count_race1_lower_bound += row['Race1']
                count_race2_positive_lower_bound += row['Race2 Positive']
                count_race2_lower_bound += row['Race2']

        try: 
            # Calculate lower bound
            Max_E1 = round((count_race1_positive_upper_bound/ count_race1_lower_bound), 4)       
        except ZeroDivisionError:
            Max_E1 = "undefined"
            pass
        try: 
            # Calculate lower bound
            Min_E1 = round((count_race1_positive_lower_bound/ count_race1_upper_bound), 4)       
        except ZeroDivisionError:
            Min_E1 = "undefined"
            pass

        try: 
            # Calculate upper bound
            Max_E2 = round((count_race2_positive_upper_bound/ count_race2_lower_bound), 4)      
        except ZeroDivisionError:
            Max_E2 = "undefined"
            pass
        try: 
            # Calculate upper bound
            Min_E2 = round((count_race2_positive_lower_bound/ count_race2_upper_bound), 4)      
        except ZeroDivisionError:
            Min_E2 = "undefined"
            pass

        # Calculate SPD for lower and upper bounds
        if Max_E1 != "undefined" and Min_E2 != "undefined":
            spd_upper = round(Max_E1 - Min_E2, 4)
        else:
            spd_upper = "undefined" 
        if Min_E1 != "undefined" and Max_E2 != "undefined":
            spd_lower = round(Min_E1 - Max_E2, 4)
        else:
            spd_lower= "undefined"


        if spd_lower != "undefined" and spd_upper != "undefined":
            if 0.0 <= spd_lower and spd_upper <= 0.2:
                satisfied = {
                    "condition1": income_range, 
                    "condition2": num_children_range,
                    "SPD": [spd_lower, spd_upper],  # Store as a list
                    "Range Satisfaction": "Full"
                }  
            elif (0.0 <= spd_lower <= 0.2 and 0.2 < spd_upper) or (spd_lower <= 0.0 and 0.0 < spd_upper <= 0.2):
                satisfied = {
                    "condition1": income_range,
                    "condition2": num_children_range,
                    "SPD": [spd_lower, spd_upper],  # Store as a list
                    "Range Satisfaction": "Partial"
                } 
           
        return satisfied, agg_counter
    



        
     


