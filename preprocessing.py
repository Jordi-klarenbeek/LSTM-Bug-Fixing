from proc_util import sequentializer
import pandas as pd
import numpy as np

# Path to the AST's
ast_before_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/before/asts.csv"
ast_after_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/asts.csv"
seq_before_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/before/seq.csv"
seq_after_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/seq.csv"
edits_path = "C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/antlr_output/after/edits.csv"

seq = sequentializer()

#reader_before = pd.read_csv(ast_before_path, sep=',', chunksize=1000)
#reader_after = pd.read_csv(ast_after_path, sep=',', chunksize=1000)
#reader_after = pd.read_csv(edits_path, sep=',', chunksize=1000)

reader_before = pd.read_csv(ast_before_path, sep=',', chunksize=1000)
reader_after = pd.read_csv(edits_path, sep=',', chunksize=1000)

before = pd.DataFrame(columns=['id', 'before'])
after = pd.DataFrame(columns=['id', 'after'])

print("load before ast trees and process")
for chunk in reader_before:
    before = seq.load_ast(before, chunk, False, "before")

#print("load after ast trees and sequentialise")
#for chunk in reader_after:
#    after = seq.load_ast(after, chunk, True, "after")

#for chunk in reader_before:
#    before = seq.load_seq(before, chunk, "before", limit=1000)

for chunk in reader_after:
    after = seq.load_seq(after, chunk, "after", limit=1000)

# Find max sequence length and pad all sequences to that length
#max_depth_before, avg_depth_before  = seq.get_max_depth(before, histogram=True)
max_length_after, avg_length_after = seq.get_seq_length(after, 'after', histogram=True)
#print(f'Max depth before: {max_depth_before}, Avg depth before: {avg_depth_before}')
print(f'Max length after: {max_length_after}, Avg length after: {avg_length_after}, Number of edit sequences : {after.shape[0]}')

max_length = 1000

after = seq.pad_sequences(after, max_length, 'after')
#before = seq.pad_sequences(before, max_length, 'before')

print("zip before trees and after sequences")
seq_path = seq.zip_df(before, after, before_tree=True)

#for i, row in seq_path.iterrows():
    # or row['seq after']==[1, 3, 2] or row['seq after']==[1, 3, 14, 10, 11, 7, 2] or row['seq before']==[1, 3, 2] or row['seq before']==[1, 3, 14, 10, 11, 7, 2]
    #len(row['before'])>max_length or
#    if len(row['before'])>max_length:
#        seq_path.drop(i, inplace=True)

seq_path.to_csv('C:/Users/jordi/OneDrive/Documenten/Master/Eind Project/Data/tree2edit.csv', index=False, mode='w')