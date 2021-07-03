import ast
import json
import torch
import pandas as pd
import numpy as np
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def string2list(s):
    return ast.literal_eval(s)

def convert_tree_to_tensors(tree_dict):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    features = tree_dict['features']
    node_order = tree_dict['node_order']
    adjacency_list = tree_dict['adjacency_list']
    edge_order = tree_dict['edge_order']

    return {
        'features': torch.tensor(features, device=device, dtype=torch.int32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }

# Import csv with before and after sequences.
def import_data(split, dataset_path, encoder, decoder):
    df_bugfixpairs = pd.read_csv(dataset_path, encoding="utf-8")
    split_index = math.floor(df_bugfixpairs.shape[0] * split)
    print(f"Dataset is split on index : {split_index}")

    df_shuffled = df_bugfixpairs.sample(frac=1).reset_index(drop=True)

    if decoder == "clas":
        # Fill list with before values
        X_total = []
        for index, row in df_shuffled.iterrows():
            # Prepare the before data different based on the encoder
            if encoder == "tree":
                tree_dict = json.loads(row['tree'])
                X = convert_tree_to_tensors(tree_dict)
            else:
                X = np.array(string2list(row['tree']))

            X_total.append(X)

        split_index = 20
        X_train = X_total[:split_index]
        X_test = X_train
        #X_test = X_total[split_index:]

        Y_total = []
        for index, row in df_shuffled.iterrows():
            Y_total.append(row['label'])

        Y_train = Y_total[:split_index]
        Y_test = Y_train
        #Y_test = Y_total[split_index:]
    else:
        # Fill list with before values
        X_total = []
        for index, row in df_shuffled.iterrows():
            # Prepare the before data different based on the encoder
            if encoder == "tree":
                tree_dict = json.loads(row['before'])
                X = convert_tree_to_tensors(tree_dict)
            else:
                X = np.array(string2list(row['before']))

            X_total.append(X)

        split_index = 20
        X_train = X_total[:split_index]
        X_test = X_train
        #X_test = X_total[split_index:]

        Y_total = []
        for index, row in df_shuffled.iterrows():
            arrayList = np.array(string2list(row['after']))
            Y_total.append(arrayList)

        Y_train = Y_total[:split_index]
        Y_test = Y_train
        #Y_test = Y_total[split_index:]

    return X_train, Y_train, X_test, Y_test


# def showPlot(points):
#    plt.figure()
#    fig, ax = plt.subplots()
# this locator puts ticks at regular intervals
#    loc = ticker.MultipleLocator(base=0.2)
#    ax.yaxis.set_major_locator(loc)
#    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def list_lengths(tensor):
    batch_size = tensor.shape[0]
    tensor_length = tensor.shape[1]
    lengths = []

    for ai in range(batch_size):
        for bi in range(tensor_length):
            if tensor[ai][bi].item() == 0:
                lengths.append(bi+1)
                break
            elif bi == tensor_length-1:
                lengths.append(bi+1)

    return lengths