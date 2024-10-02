from itertools import product
from turtle import distance
from Manhattan_distance import Manhattan_distance
from itertools import product

class predicatesPossibleValues:

    def calculate_combinations(self, predicates):
        num_combinations = 1
        for predicate in predicates:
            num_combinations *= len(predicate['values'])
        return num_combinations
    '''
    def numerical_att_dist(self, refinement, user_predicate_list):
        # Simplified example of calculating the Manhattan distance for two numerical attributes
        distance = 0
        for user_pred in user_predicate_list:
            for pred in refinement:
                if pred['column'] == user_pred['column']:
                    distance += abs(pred['value'] - user_pred['value'])
        return distance
    '''
    def numerical_att_dist(self, possible_refinements, user_predicate_list):
        # Example Manhattan distance calculation
        updated_refinements = []
        
        for refinement in possible_refinements:
            # Ensure that refinement is a dictionary
            if isinstance(refinement, dict):
                distance = 0
                # Iterate through each predicate in the refinement and compare with user predicates
                for pred in refinement.get('refinements', []):  # Ensures 'refinements' key exists
                    for user_pred in user_predicate_list:
                        if pred['column'] == user_pred['column']:
                            distance += self.calculate(pred['value'], user_pred['value'])
                # Create a new dictionary including the distance
                refinement_with_distance = {
                    **refinement,  # Spread the existing refinement dictionary
                    'distance': distance  # Add the new distance key
                }
                updated_refinements.append(refinement_with_distance)
            else:
                raise TypeError("Refinement should be a dictionary")

        return updated_refinements

    def calculate(self, value1, value2):
        return abs(value1 - value2)

    def generate_possible_refinments_similarity(self, all_pred_possible_values, UserpredicateList):
        possible_refinements = []

        # Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': pred_info['values']} for pred_info in all_pred_possible_values]

        # Generate all combinations of values for all predicates
        combinations = product(*[predicate['values'] for predicate in predicates])

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


        return sorted_possible_refinements




    def generate_possible_refinments_inorder(self, all_pred_possible_values, UserpredicateList):
        possible_refinements = []

        # Extract information about each predicate
        predicates = [{'predicate': pred_info['predicate'], 'values': pred_info['values']} for pred_info in all_pred_possible_values]

        # Generate all combinations of values for all predicates
        combinations = list(product(*[predicate['values'] for predicate in predicates]))

        # Construct possible refinements based on combinations
        for combination in combinations:
            refinement_info = []
            for pred_info, val in zip(predicates, combination):
                predicate_info = pred_info['predicate']
                # Create a dictionary with column, operator, and value
                single_refinement = {
                    'column': predicate_info['column'],
                    'operator': predicate_info['operator'],
                    'value': val
                }
                refinement_info.append(single_refinement)
            possible_refinements.append(refinement_info)

        # Sort the possible refinements by column, operator, and value
        sorted_possible_refinements = sorted(possible_refinements, key=lambda x: [(entry['column'], entry['operator'], entry['value']) for entry in x])

        return sorted_possible_refinements


