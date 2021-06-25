# LSTM-Bug-Fixing
Bug fixer using LSTM models

The preprocessing is applied with preprocessing.py and proc_util.py. The files necessary for the preprocessing are generating with ANTLR and the Big-vul dataset.

The lstm model is used with the command line by calling main.py with the following arguments:
-enc tree | seq
-dec seq | seq_att
-csv "path/to/file.csv"
-b batch_size 
-e number of epochs 
-hs hidden vector size
-l learning rate
-end end token for decoder
-begin begin token for decoder

For example:
python main.py -enc tree -dec seq -csv "~\Data\tree2edit.csv" -b 32 -e 2 -hs 256 -l 0.001 -end 130 -begin 129
