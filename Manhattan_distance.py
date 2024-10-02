
class Manhattan_distance:

    def manhattan_distance(self, x1, y1, x2, y2):
        # Convert x2 and y2 to integers if they are strings
        print(x2)
        x2 = int(x2) if isinstance(x2, str) else x2
        y2 = int(y2) if isinstance(y2, str) else y2

        # Ensure x1 and y1 are also integers
        x1 = int(x1) if isinstance(x1, str) else x1
        y1 = int(y1) if isinstance(y1, str) else y1

        return abs(x1 - x2) + abs(y1 - y2)

    def numerical_att_dist(self, possibleRefinments, UserpredicateList):
        distances = []
        values1= []
        values2 = []
        
        # Extracting values from the dictionaries

        values1 = [first for first, _ in possibleRefinments]
        values2 = [second for _, second in possibleRefinments]


        # Extracting threshold values from the range list
        income_threshold = [r[2] for r in UserpredicateList if r[0] == 'WKHP'][0]
        num_child_threshold = [r[2] for r in UserpredicateList if r[0] == 'SCHL'][0]

        # Calculate Manhattan distance for each pair of values
        #print("***** Distance for possiblePredicates: ")
        for income, num_child in zip(values1, values2):
            print(income, num_child, income_threshold, num_child_threshold)  
            distance = self.manhattan_distance(income, num_child, income_threshold, num_child_threshold)   
                    
            #print(f"Manhattan distance for income={gpa} and num-children={age} = {round(distance, 2)}")
            distances.append({'income': income, 'operator': '>=', 'distance': distance})
        return distances
        




        
