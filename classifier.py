from sklearn.ensemble import RandomForestClassifier

class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, feature_names=None, n_jobs=1, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.feature_names_mapping = self._build_feature_names_mapping()

    def _build_feature_names_mapping(self):

        if self.feature_names is None:
            raise ValueError("Feature names are not provided.")

        feature_names_mapping = {i: name for i, name in enumerate(self.feature_names)}
        return feature_names_mapping
    
    
    def get_feature_name_from_node(self, feature_index):
        """Get the feature name based on the feature index."""
        if feature_index == -2:
            return "Leaf Node"
        elif feature_index in self.feature_names_mapping:
            return self.feature_names_mapping[feature_index]
        else:
            raise ValueError(f"Feature index {feature_index} not found in feature names mapping.")

    def update_feature_names_mapping(self, feature_offset):
        if self.feature_names is None:
            raise ValueError("Feature names are not provided.")

        if self.feature_names_mapping is None:
            self.feature_names_mapping = dict(enumerate(self.feature_names))

        # Shift the keys in the mapping by the feature offset
        shifted_mapping = {key + feature_offset: value for key, value in self.feature_names_mapping.items()}

        # Update the mapping
        #self.feature_names_mapping.update(shifted_mapping)
        self.feature_names_mapping = shifted_mapping
