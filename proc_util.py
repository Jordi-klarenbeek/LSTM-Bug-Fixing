import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from treelstm import calculate_evaluation_orders

"""
Sequentializer object:
Creates a sequence of a JSON ast tree
Add padding 
Measure max length of sequences and depth of trees
"""
class sequentializer:
    def __init__(self):
        self.depth_list = []

    # Use depth-first algorithm to create a sequence of the tree
    # With s sequence style summarization of tree
    def sequentialize(self, ast_dict):
        # initialize list with start token
        self.ast_list = []

        # Add open bracket token
        #self.ast_list.append(1)

        # Append root token
        self.ast_list.append(ast_dict['token'])

        # Loop through the children and append them to ast_list
        if 'children' in ast_dict.keys():
            self.dftraversal(ast_dict)

        # Add close bracket token
        #self.ast_list.append(2)

        return self.ast_list

    # Traverse Depth-First over tree dictionary
    def dftraversal(self, ast_dict):
        for child in ast_dict['children']:
            # Select only reserved tokens if reduced is true
            self.ast_list.append(child['token'])

            # Only recurse if the node has children
            if 'children' in child.keys():
                self.dftraversal(child)

    # Pad sequences with zeros until they are same length as max length in dataset
    def pad_sequences(self, seq_df, length, column_name):
        for i, row in seq_df.iterrows():
            while len(row[column_name]) < length:
                row[column_name].append(0)

        return seq_df

    # combine the before and after dataframe
    def zip_df(self, before_df, after_df, before_tree):
        df = pd.DataFrame(columns=['id', 'before', 'after'])

        # iterate over every row in before_df dataframe
        for i, row in before_df.iterrows():
            # program id row['id']
            if row['id'] in after_df.values:
                id = row['id']
                if before_tree:
                    before = json.dumps(row['before'])
                else:
                    before = row['before']

                after = after_df.loc[after_df['id'] == row['id']]['after'].tolist()[0]

                df = df.append({'id': id, 'before': before, 'after': after}, ignore_index=True)

        return df

    # recursively write depth of node in dictionary and return depths if terminal nodes are reached
    def depth(self, ast_dict, parent_depth):
        ast_dict['depth'] = parent_depth + 1

        if 'children' in ast_dict.keys():
            for child in ast_dict['children']:
                self.depth(child, ast_dict['depth'])
        else:
            self.depth_list.append(ast_dict['depth'])

    # Calculate depths of all trees and return max and average
    def get_max_depth(self, ast_df, histogram):
        depths = list()

        for i, row in ast_df.iterrows():
            self.depth(row['tree'], 0)
            depths.append(max(self.depth_list))
            self.depth_list = []

        if histogram:
            n, bins, patches = plt.hist(depths, bins=100)
            plt.xlabel("Depths")
            plt.ylabel("Frequency")
            plt.title("Histogram")
            plt.show()

        return max(depths), sum(depths)/len(depths)

    # input     dictionary with format {id: [ast sequence]}
    # output    max length of ast sequences
    def get_seq_length(self, seq_df, column_name, histogram):
        lengths = list()

        for i, row in seq_df.iterrows():
            lengths.append(len(row[column_name]))

        if histogram:
            n, bins, patches = plt.hist(lengths, bins=100)
            plt.xlabel("Lengths")
            plt.ylabel("Frequency")
            plt.title("Histogram")
            plt.show()

        return max(lengths), sum(lengths)/len(lengths)

    def load_ast(self, seq_df, ast, sequentialize, column_name):
        for index, row in ast.iterrows():
            ast_dict = json.loads(row['AST'])
            if(sequentialize):
                ast_list = self.sequentialize(ast_dict)
            else:
                #ast_list = ast_dict
                ast_list = self.create_ast_lists(ast_dict)

            seq_df = seq_df.append([{'id' : row['id'],  column_name : ast_list}], ignore_index=True)

        return seq_df

    def load_seq(self, seq_df, edit, column_name, limit):
        for index, row in edit.iterrows():
            edit_list = json.loads(row['Edits'])

            if len(edit_list) <= limit:
                seq_df = seq_df.append([{'id': row['id'], column_name: edit_list}], ignore_index=True)

        return seq_df

    def create_ast_lists(self, tree):
        self.adj_list = []
        self.features = []
        self.index = 0

        # Recursively calculate the adjacency list and features list
        # pass -1 for the "parent index" of the root node
        self.traverse_and_list(tree, -1)

        node_order, edge_order = calculate_evaluation_orders(self.adj_list, len(self.features))

        return {
            "features": self.features,
            "node_order": node_order.tolist(),
            "adjacency_list": self.adj_list,
            "edge_order": edge_order.tolist()
        }

    # Traverse Depth-First over tree dictionary and fill features list and adjacency list
    def traverse_and_list(self, tree, parent_index):
        tree['index'] = self.index
        self.features.append([tree['token']])
        self.index += 1

        if parent_index != -1:
            self.adj_list.append([parent_index, tree['index']])

        # Only recurse if the node has children
        if 'children' in tree.keys():
            for child in tree['children']:
                self.traverse_and_list(child, tree['index'])