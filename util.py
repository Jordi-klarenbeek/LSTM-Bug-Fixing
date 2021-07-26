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
    print(f"Number of datapoints: {df_bugfixpairs.shape[0]}")
    df_shuffled = df_bugfixpairs.sample(frac=1).reset_index(drop=True)

    if decoder == "clas":
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
            Y_total.append(row['label'])

        Y_train = Y_total[:split_index]
        Y_test = Y_train
        #Y_test = Y_total[split_index:]

        # save extra info in dataset
        test_extra_info = df_shuffled.loc[split_index:, 'CWE ID':'Vulnerability Classification']
        test_extra_info['CWE ID'] = test_extra_info['CWE ID'].fillna('CWE-000')
        test_extra_info = test_extra_info.reset_index(drop=True)
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

        # save extra info in dataset
        test_extra_info = df_shuffled.loc[split_index:, 'CWE ID':'Vulnerability Classification']
        test_extra_info = test_extra_info.reset_index(drop=True)

    return X_train, Y_train, X_test, Y_test, test_extra_info


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

def save_best_10_programs(output_df):
    top10 = output_df.sort_values('BLEU score', ascending=False).head(n=10)

    print(top10)

    top10.to_csv('top_10_program_results.csv')

def print_CWE_average(output_df):
    output_df['CWE ID'] = output_df['CWE ID'].fillna('CWE-000')
    CWE_average = pd.DataFrame(columns=['CWE', 'sum BLEU score', 'average BLEU score', 'amount'])

    for i, row in output_df.iterrows():
        cwe_id = row['CWE ID']

        if cwe_id not in CWE_average.CWE.values:
            CWE_average = CWE_average.append([{'CWE': cwe_id, 'sum BLEU score': row['BLEU score'],
                                               'average BLEU score': row['BLEU score'], 'amount': 1}], ignore_index=True)
        else:
            # Select row with CWE ID and calc new average of that CWE ID
            old_average = CWE_average.loc[CWE_average['CWE'] == cwe_id]

            new_sum = row['BLEU score'] + old_average['sum BLEU score'].iloc[0]
            new_amount = old_average['amount'].iloc[0] + 1
            new_avg = (row['BLEU score'] + old_average['sum BLEU score'].iloc[0]) / new_amount

            CWE_average.loc[CWE_average['CWE'] == cwe_id, 'sum BLEU score'] = new_sum
            CWE_average.loc[CWE_average['CWE'] == cwe_id, 'average BLEU score'] = new_avg
            CWE_average.loc[CWE_average['CWE'] == cwe_id, 'amount'] = new_amount

    print(CWE_average.sort_values('average BLEU score', ascending=False))

def calc_clas_CWE_accuracy(output_df, cwe_id, prediction, label):
    CWE_accuracy_df = output_df

    if round(prediction.item()) == label.item():
        correct = 1
    else:
        correct = 0

    if cwe_id not in CWE_accuracy_df.CWE.values:
        CWE_accuracy_df = CWE_accuracy_df.append([{'CWE': cwe_id, 'match': correct,
                                          'total': 1, 'accuracy': correct}], ignore_index=True)
    else:
        # Select row with CWE ID and calc new average of that CWE ID
        old_average = CWE_accuracy_df.loc[CWE_accuracy_df['CWE'] == cwe_id]

        new_match = old_average['match'].iloc[0] + correct
        new_total = old_average['total'].iloc[0] + 1
        new_accuracy = new_match / new_total

        CWE_accuracy_df.loc[CWE_accuracy_df['CWE'] == cwe_id, 'match'] = new_match
        CWE_accuracy_df.loc[CWE_accuracy_df['CWE'] == cwe_id, 'total'] = new_total
        CWE_accuracy_df.loc[CWE_accuracy_df['CWE'] == cwe_id, 'accuracy'] = new_accuracy

    return CWE_accuracy_df

def translate(tokens, input_form, vocab_path, end_token):
    vocab = json.load(open(vocab_path))
    translated_tokens = []
    keys = list(vocab.keys())
    vals = list(vocab.values())

    if input_form == "tensor":
        for token in tokens:
            if token.item() == 0:
                break
            translated_tokens.append(keys[vals.index(token.item())])
            if token.item() == end_token:
                break
    else:
        for token in tokens:
            if token == 0:
                break
            translated_tokens.append(keys[vals.index(token)])
            if token == end_token:
                break

    return translated_tokens