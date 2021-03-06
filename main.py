from __future__ import unicode_literals, print_function, division
import torch
from torch import optim
import torch.nn as nn
import util
import metrics
import model
import math
import argparse
import training

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'The model will use as device : {device} | With {torch.get_num_threads()} threads')

    args_parser = argparse.ArgumentParser(
        description='AST parser can create ASTs from CPP code in JSON format and can transform the AST trees back to code')

    args_parser.add_argument('-enc', '--encoder',
                             metavar='encoder',
                             type=str,
                             help='The architecture of the encoder',
                             default='')

    args_parser.add_argument('-dec', '--decoder',
                             metavar='decoder',
                             type=str,
                             help='The architecture of the decoder',
                             default='')

    args_parser.add_argument('-csv', '--csv_file_path',
                             metavar='csv_file_path',
                             type=str,
                             help='the path to the CSV file containing data of before and after programs',
                             required=True)

    args_parser.add_argument('-b', '--batch_size',
                             metavar='batch_size',
                             type=str,
                             help='the batch size of the model',
                             required=True)

    args_parser.add_argument('-e', '--epochs',
                             metavar='epochs',
                             type=str,
                             help='the number of epochs the model trains',
                             required=True)

    args_parser.add_argument('-hs', '--hidden_size',
                             metavar='hidden_size',
                             type=str,
                             help='the dimension size of the hidden vector and word embeddings',
                             required=True)

    args_parser.add_argument('-l', '--learning_rate',
                             metavar='learning_rate',
                             type=str,
                             help='the learning rate of the model',
                             required=True)

    args_parser.add_argument('-len', '--sequence_length',
                             metavar='sequence_length',
                             type=str,
                             help='the length of the sequences',
                             required=True)

    args_parser.add_argument('-end', '--end_token',
                             metavar='end_token',
                             type=str,
                             help='the end token of a sequence',
                             required=True)

    args_parser.add_argument('-begin', '--begin_token',
                             metavar='begin_token',
                             type=str,
                             help='the begin token of a sequence',
                             required=True)

    args_parser.add_argument('-o', '--output_programs',
                             metavar='output_programs',
                             type=str,
                             help='save the top ten best produced programs in text file',
                             required=True)

    args_parser.add_argument('-vocab', '--output_vocab',
                             metavar='output_vocab',
                             type=str,
                             help='the vocabulary used to translate tokens to concrete AST',
                             required=True)

    args = args_parser.parse_args()

    encoder_arch = args.encoder
    decoder_arch = args.decoder
    training_split = 0.9

    print(f"Path to input data : {args.csv_file_path}")
    X_train, Y_train, X_test, Y_test, vuln_info = util.import_data(training_split, args.csv_file_path, encoder_arch, decoder_arch)

    # Fill in hyperparameters
    teacher_forcing_ratio = 0.9
    learning_rate = float(args.learning_rate)
    hidden_size = int(args.hidden_size)
    vocab_size = 179
    batch_size = int(args.batch_size)
    n_iter = math.floor(len(X_train) / batch_size)
    max_length = int(args.sequence_length)
    num_layers = 2
    bidirectional = True
    epochs = int(args.epochs)
    end_token = int(args.end_token)
    begin_token = int(args.begin_token)
    early_stopping = True
    min_epochs = 25
    vocab_path = args.output_vocab
    output_programs = args.output_programs

    print(f"Hidden size : {hidden_size} | With zero padding cutoff")

    if encoder_arch == "seq":
        encoder = model.SeqEncoderLSTM(vocab_size, hidden_size, batch_size, num_layers, bidirectional).to(device)
    elif encoder_arch == "tree":
        encoder = model.ChildSumTreeLSTM(vocab_size, hidden_size).to(device)
        num_layers = 1
    else:
        raise Exception("Select valid decoder architecture")

    if decoder_arch == "seq":
        decoder = model.SeqDecoderLSTM(hidden_size, vocab_size, batch_size, num_layers).to(device)
    elif decoder_arch == "seq_att":
        if encoder_arch != "seq":
            raise Exception("Sequential decoder with attention is not possible with a non sequential encoder")
        else:
            decoder = model.SeqDecoderAttentionLSTM(hidden_size, vocab_size, batch_size, max_length, num_layers,
                                                    dropout_p=0.1).to(device)
    elif decoder_arch == "clas":
        decoder = model.BinaryClassifierNet(hidden_size, num_layers).to(device)
    else:
        raise Exception("Select valid encoder architecture")

    # give sum of parameters to the optimizer
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate)

    if decoder_arch == "clas":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()

    """"
    # Test the baseline BLEU score before transformation
    bleuScores = []
    for i in range(len(Y_test)):
        bleuScores.append(metrics.calcBleu(X_test[i], Y_test[i], end_token))

    meanScores = sum(bleuScores) / len(bleuScores)
    varScores = sum([((x - meanScores) ** 2) for x in bleuScores]) / len(bleuScores)
    stdScores = varScores ** 0.5

    print(f'The baseline average bleu score is : {meanScores}')
    print(f'The baseline std of the bleu score is : {stdScores}')
    """

    modelTrainer = training.Trainer(encoder, decoder, optimizer, criterion, batch_size,
                                    max_length, teacher_forcing_ratio, device, begin_token, end_token, vocab_size)

    print( f"Training data: {len(X_train)}, Test data: {len(X_test)}")

    loss_epochs = []
    for i in range(epochs):
        print(f'+++ Epoch number : {i} +++')
        loss = modelTrainer.trainIters(X_train, Y_train, n_iter, print_every=10)

        print(f'+++ Average Epoch loss : {loss} +++')

        # Check for early stopping if the training takes longer than the minimal amount of epochs
        if i > min_epochs and early_stopping:

            # Calculate average loss of last 5 epochs
            avg_loss = sum(loss_epochs[i-5:i])/5

            # Stop training if the difference between this epoch's loss and avg loss is 1%
            if abs(avg_loss-loss)<(avg_loss*0.05):
                print(f'No significant difference in loss, training is stopped...')
                break

        loss_epochs.append(loss)

    if decoder_arch == "clas":
        modelTrainer.testClas(X_test, Y_test, vuln_info)
    elif encoder_arch == "tree":
        modelTrainer.testtree(X_test, Y_test, output_programs, vocab_path, vuln_info)
    else:
        modelTrainer.testseq(X_test, Y_test, output_programs, vocab_path, vuln_info)

if __name__ == "__main__":
    main()