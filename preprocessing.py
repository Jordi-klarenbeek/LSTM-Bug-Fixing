from proc_util import preprocessor
import pandas as pd
import numpy as np
import json

# Path to the AST's
ast_before_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/before/asts.csv"
ast_after_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/asts.csv"
seq_before_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/before/seq.csv"
seq_after_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/seq.csv"
edit_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/edits.csv"

proc = preprocessor()

reader_before = pd.read_csv(ast_before_path, sep=',', chunksize=1000)
reader_after = pd.read_csv(ast_after_path, sep=',', chunksize=1000)

before = pd.DataFrame(columns=['id', 'CWE ID', 'CVE ID', 'Vulnerability Classification', 'before', 'label'])
after = pd.DataFrame(columns=['id', 'CWE ID', 'CVE ID', 'Vulnerability Classification', 'after'])

print("load before ast trees and process")
for chunk in reader_before:
    before = proc.load_ast(before, chunk, True, "before", 1)

print("load after ast trees and sequentialise")
for chunk in reader_after:
    before = proc.load_ast(before, chunk, True, "before", 0)

#for chunk in reader_before:
#    before = proc.load_seq(before, chunk, "tree", limit=1000)

#for chunk in reader_after:
#    after = proc.load_seq(after, chunk, "after", limit=700)

# Find max sequence length and pad all sequences to that length
#max_depth_before, avg_depth_before  = proc.get_max_depth(before, histogram=True)
#max_length_after, avg_length_after = proc.get_seq_length(after, 'after', histogram=True)
#print(f'Max depth before: {max_depth_before}, Avg depth before: {avg_depth_before}')
#print(f'Max length seq: {max_length_after}, Avg length: {avg_length_after}, Number of sequences : {after.shape[0]}')

max_length = 1000

before = proc.pad_sequences(before, max_length, 'before')
#after = proc.pad_sequences(after, max_length, 'after')

#print("zip before trees and after sequences")
#seq_path = proc.zip_df(before, after)

for i, row in before.iterrows():
    # or row['seq after']==[1, 3, 2] or row['seq after']==[1, 3, 14, 10, 11, 7, 2] or row['seq before']==[1, 3, 2] or row['seq before']==[1, 3, 14, 10, 11, 7, 2]
    #len(row['before'])>max_length or
    if len(row['before'])>max_length:
        before.drop(i, inplace=True)

#max_length_after, avg_length_after = seq.get_seq_length(seq_path, 'after', histogram=True)
#print(f'Max length seq: {max_length_after}, Avg length: {avg_length_after}, Number of sequences : {seq_path.shape[0]}')

before.to_csv('C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/binseq_extrainfo.csv', index=False, mode='w')