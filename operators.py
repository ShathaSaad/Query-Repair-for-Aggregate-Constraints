

class operators:

    def apply_operator(self, Min_value, Max_value, condition_value, operator, type):
        # Define a dictionary mapping operators to their respective lambda functions
        operators = {
            '<': lambda x, y: x < y,
            '<=': lambda x, y: x <= y,
            '>': lambda x, y: x > y,
            '>=': lambda x, y: x >= y,
            '==': lambda x, y, z: x == z and y == z,
            '!=': lambda x, y: x != y,
        }

        operators_partial = {
            '==': lambda x_min, x_max, y: (x_min == y and x_max != y) or (x_min != y and x_max == y), #one of min and max is == and the other is not
            '!=': lambda x_min, x_max, y: (x_min != y and x_max == y) or (x_min == y and x_max != y)  #one of min and max is != and the other is ==
        }
    
        # Check if the operator is valid and apply the operation
        if operator == '>=' and type == "Full":
            return operators[operator](Min_value, condition_value)
        elif operator == '<=' and type == "Full":
            return operators[operator](Max_value, condition_value)
        elif operator == '>' and type == "Full":
            return operators[operator](Min_value, condition_value)
        elif operator == '<' and type == "Full":
            return operators[operator](Max_value, condition_value)
        elif operator == '==' and type == "Full":
            return operators[operator](Min_value, Max_value, condition_value)
        elif operator == '!=' and type == "Full":
            return operators[operator](Min_value, Max_value, condition_value)
        
        if operator == '>=' and type == "Partial":
            return operators[operator](Max_value, condition_value)
        elif operator == '<=' and type == "Partial":
            return operators[operator](Min_value, condition_value)
        elif operator == '>' and type == "Partial":
            return operators[operator](Max_value, condition_value)
        elif operator == '<' and type == "Partial":
            return operators[operator](Min_value, condition_value)

        elif operator == '==' and type == "Partial":
            return operators_partial[operator](Min_value, Max_value, condition_value)
        elif operator == '!=' and type == "Partial":
            return operators_partial[operator](Min_value, Max_value, condition_value)

        else:
            raise ValueError("Unsupported operator")


    def apply_operator_ranges(self, Min_value, Max_value, Min_condition, Max_condition, operator, type):
        # Define a dictionary mapping operators to their respective lambda functions
        operators = {
            '<': lambda x, y: x < y,
            '<=': lambda x, y: x <= y,
            '>': lambda x, y: x > y,
            '>=': lambda x, y: x >= y,
            '==': lambda x_min, x_max, y_min, y_max: x_min == y_min and x_max == y_max,
            '!=': lambda x_min, x_max, y_min, y_max: x_min != y_min and x_max != y_max,
        }
        operators_partial = {
            '==': lambda x_min, x_max, y_min, y_max: (x_min == y_min and x_max != y_max) or (x_min != y_min and x_max == y_max), #one of min and max is == and the other is not
            '!=': lambda x_min, x_max, y_min, y_max: (x_min != y_min and x_max == y_max) or (x_min == y_min and x_max != y_max)  #one of min and max is != and the other is ==
        }
    
        # Check if the operator is valid and apply the operation
        if operator == '>=' and type == "Full":
            return operators[operator](Min_value, Max_condition)
        elif operator == '<=' and type == "Full":
            return operators[operator](Max_value, Min_condition)
        elif operator == '>' and type == "Full":
            return operators[operator](Min_value, Max_condition)
        elif operator == '<' and type == "Full":
            return operators[operator](Max_value, Min_condition)
        elif operator == '==' and type == "Full":
            return operators[operator](Min_value, Max_value, Min_condition, Max_condition)
        elif operator == '!=' and type == "Full":
            return operators[operator](Min_value, Max_value, Min_condition, Max_condition)

        if operator == '>=' and type == "Partial":
            return operators[operator](Max_value, Min_condition)
        elif operator == '<=' and type == "Partial":
            return operators[operator](Min_value, Max_condition)
        elif operator == '>' and type == "Partial":
            return operators[operator](Max_value, Min_condition)
        elif operator == '<' and type == "Partial":
            return operators[operator](Min_value, Max_condition)
        elif operator == '==' and type == "Partial":
            return operators_partial[operator](Min_value, Max_value, Min_condition, Max_condition)
        elif operator == '!=' and type == "Partial":
            return operators_partial[operator](Min_value, Max_value, Min_condition, Max_condition)

        else:
            raise ValueError("Unsupported operator")


 
    
    
