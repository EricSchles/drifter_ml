"""
Note: While these tests fulfill the necessity condition,
they likely do not fulfill the sufficiency condition.

I'm not sure what a notion of sufficiency would even look like.
"""

def manhattan_distance(a, b):
    return abs(a - b)

class ClassificationBiasTests:
    def __init__(self,
                 clf,
                 train_data,
                 test_data,
                 target_name,
                 column_names,
                 protected_variables):
        self.clf = clf
        self.train_data = train_data
        self.test_data = test_data
        self.column_names = column_names
        self.target_name = target_name
        self.y_test = test_data[target_name]
        self.X_test = test_data[column_names]
        self.y_train = train_data[target_name]
        self.X_train = train_data[column_names]
        self.train = train_data
        self.test = test_data
        self.pv = protected_variables
        self.pv_train = train_data[protected_variables]
        self.pv_test = test_data[protected_variables]
        self.classes = set(self.y_train)

    def data_bias_upper_boundary(self,
                                 upper_boundary: dict,
                                 distance_function=manhattan_distance):
        """
        This function compares the protected variable groupings
        against the proportions of each class for the target variable.
        If any groupings exceed the boundary per class, then the
        test has failed.
        
        Parameters
        ----------
        upper_boundary : dict - a dictionary of upper boundaries
        for each class in the target variable.
        distance_function : callable - a distance metric
        used to measure the distance between the proportions per 
        group per class.

        Returns
        -------
        A dictionary of comparisons between groups.
        dictionary key - (the group of interest, the class of target variable)
        dictionary values:
          * comparison_group - the group being compared
          * distance - the distance between the groups for the given class
          * within_boundary - whether or not the test passed (and the 
            target variable class was close enough.
        """
        group_label_proportions = {}
        group_klass_results = {}
        for group, tmp in self.train.groupby(self.pv):
            group_label_proportions[group] = tmp[self.target_name].value_counts()/tmp.shape[0]
        for group_one in group_label_proportions:
            for group_two in group_label_proportions:
                if group_one == group_two:
                    continue
                for klass in upper_boundary:
                    distance = distance_function(
                        group_label_proportions[group_one][klass],
                        group_label_proportions[group_two][klass]
                    )
                    result = distance < upper_boundary[klass]
                    group_klass_results[(group_one, klass)] = {
                        "comparison_group": group_two,
                        "distance": distance,
                        "within_boundary": result
                    }
        return group_klass_results
