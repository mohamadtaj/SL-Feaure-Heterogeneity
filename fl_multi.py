import numpy as np
from copy import deepcopy
from classifier import CustomRandomForestClassifier
from utils import offset_feature_indexes

class FLMulti:
    def __init__(self, nodes):

        self.nodes = nodes
        self.combined_rf = None
        self.trees = []

    def define_rf(self):
        
        """
        Combine the random forests from each node into a single forest.
        """
        
        combined_feature_names = []
        for node in self.nodes:
            combined_feature_names.extend(list(node.feature_names))

        self.combined_rf = CustomRandomForestClassifier(feature_names=combined_feature_names, n_jobs=1)

        all_estimators = []

        cumulative_offset = 0

        # For each node, offset its trees and update the mapping.
        for node in self.nodes:
            for tree in node.rf.estimators_:
                offset_feature_indexes(tree.tree_, cumulative_offset)
                all_estimators.append(tree)

            node.rf.update_feature_names_mapping(cumulative_offset)
            cumulative_offset += len(node.feature_names)            
            

        self.combined_rf.estimators_ = all_estimators

        self.combined_rf.n_estimators = len(self.combined_rf.estimators_)
        self.trees = self.combined_rf.estimators_

        all_classes = sorted(set().union(*[est.classes_ for est in [n.rf for n in self.nodes]]))
        self.combined_rf.classes_ = np.array(all_classes)
        # self.combined_rf.classes_ = self.nodes[0].rf.classes_

    def broadcast_model(self):
        """Broadcast the combined forest to every node."""
        for node in self.nodes:
            node.combined_rf = self.combined_rf
