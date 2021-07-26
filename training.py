import torch
import random
import time
import util
import metrics
import model
import math
import gc
import pandas as pd
from treelstm import batch_tree_input
from tqdm import tqdm


class Trainer(object):
    def __init__(self, encoder, decoder, optimizer, criterion, batch_size, max_length,
                 teacher_forcing_ratio, device, begin_token, end_token, vocab_size):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.begin_token = begin_token
        self.end_token = end_token
        self.vocab_size = vocab_size

    # helper function for training
    def trainTree(self, input_tree, target_tensor):
        # Store target sequence length, minus 1 since start token does not need to be predicted
        target_length = target_tensor.size(1) - 1

        # Initialize zero tensor for encoder outputs, but this will only be used if decoder attention is supported with tree encoder
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, self.encoder.hidden_size, device=self.device)

        self.optimizer.zero_grad() # maybe only one optimizer

        loss = 0

        # returns tuple with hidden state and cell state
        encoder_state = self.encoder(input_tree['features'], input_tree['node_order'], input_tree['adjacency_list'], input_tree['edge_order'])

        hidden = torch.zeros(self.decoder.num_layers, self.batch_size, self.encoder.hidden_size, device=self.device)
        cell = torch.zeros(self.decoder.num_layers, self.batch_size, self.encoder.hidden_size, device=self.device)

        # Create index list for accessing all root node hidden and cell states
        tree_indices = input_tree['tree_sizes']
        tree_indices.insert(0,0)
        # Pop the last indice as it does not reference the position of a tree roots hidden state
        tree_indices.pop()

        for layer in range(self.decoder.num_layers):
            j = 0
            for index in tree_indices:
                hidden[layer][j] = encoder_state[0][index]
                cell[layer][j] = encoder_state[1][index]
                j+=1

        decoder_hidden_cell = (hidden, cell)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # print("Training with teacher forcing...")
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    target_tensor[:, di].unsqueeze(1), decoder_hidden_cell, encoder_outputs)

                # calculate the loss separate for every output in the batch
                loss += self.criterion(decoder_output, target_tensor[:, di + 1])
        else:
            # print("Training with no teacher forcing...")
            # Initialize sequence with start token
            decoder_input = torch.tensor([self.begin_token]*self.batch_size, dtype=torch.long, device=self.device)

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    decoder_input.unsqueeze(1), decoder_hidden_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        loss.backward()

        self.optimizer.step()

        # clear cache on the GPU
        if torch.cuda.is_available():
            del encoder_state
            gc.collect()
            torch.cuda.empty_cache()

        return loss.item() / target_length

    def trainSeq(self, input_tensor, target_tensor):
        encoder_hidden_cell = self.encoder.initHidden()

        self.optimizer.zero_grad()

        # Target length is the length of the target_tensor without start token
        target_length = target_tensor.size(1) - 1

        loss = 0

        encoder_outputs, encoder_hidden_cell = self.encoder(input_tensor.view(self.batch_size, self.max_length), encoder_hidden_cell)

        # Reshape hidden cell tuple to concat bidirectional values
        hidden = encoder_hidden_cell[0].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
        cell = encoder_hidden_cell[1].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
        decoder_hidden_cell = (hidden, cell)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # print("Training with teacher forcing...")
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    target_tensor[:, di].unsqueeze(1), decoder_hidden_cell, encoder_outputs)
                # calculate the loss separate for every output in the batch
                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        else:
            # print("Training with no teacher forcing...")
            # Initialize sequence with start token
            decoder_input = torch.tensor([self.begin_token] * self.batch_size, dtype=torch.long, device=self.device)

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    decoder_input.unsqueeze(1), decoder_hidden_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                # calculate the loss separate for every output in the batch
                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        loss.backward()

        self.optimizer.step()

        # clear cache on the GPU
        if torch.cuda.is_available():
            del encoder_outputs
            gc.collect()
            torch.cuda.empty_cache()

        return loss.item() / (target_length)

    def trainClassifier(self, input_tensor, labels):
        self.optimizer.zero_grad()

        if isinstance(self.encoder, model.SeqEncoderLSTM):
            encoder_hidden_cell = self.encoder.initHidden()

            encoder_outputs, encoder_hidden_cell = self.encoder(input_tensor.view(self.batch_size, self.max_length), encoder_hidden_cell)

            # Reshape hidden cell tuple to concat bidirectional values
            hidden = encoder_hidden_cell[0].view(1, self.batch_size, self.encoder.hidden_size*self.encoder.num_layers)

            # clear cache on the GPU
            if torch.cuda.is_available():
                del encoder_outputs
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # returns tuple with hidden state and cell state
            encoder_state = self.encoder(input_tensor['features'], input_tensor['node_order'], input_tensor['adjacency_list'],
                                         input_tensor['edge_order'])

            hidden = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)

            # Create index list for accessing all root node hidden and cell states
            tree_indices = input_tensor['tree_sizes']
            tree_indices.insert(0, 0)
            # Pop the last indice as it does not reference the position of a tree roots hidden state
            tree_indices.pop()

            j = 0
            for index in tree_indices:
                hidden[0][j] = encoder_state[0][index]
                j += 1

            # clear cache on the GPU
            if torch.cuda.is_available():
                del encoder_state
                gc.collect()
                torch.cuda.empty_cache()

        prediction = self.decoder(hidden)

        loss = self.criterion(prediction, labels.view(1, self.batch_size, 1))

        loss.backward()

        self.optimizer.step()

        # clear cache on the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.item()

    def trainIters(self, X_list, Y_list, n_iters, print_every=100):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        loss_total = 0

        for iter in range(n_iters):
            if isinstance(self.encoder, model.ChildSumTreeLSTM):
                input_tree = X_list[iter * self.batch_size:(iter + 1) * self.batch_size]
                batched_trees = batch_tree_input(input_tree)

                if isinstance(self.decoder, model.BinaryClassifierNet):
                    target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                                 dtype=torch.float, device=self.device)
                    loss = self.trainClassifier(batched_trees, target_tensor)
                else:
                    target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                                 dtype=torch.long, device=self.device)
                    loss = self.trainTree(batched_trees, target_tensor)

            elif isinstance(self.encoder, model.SeqEncoderLSTM):
                input_tensor = torch.tensor(X_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                            dtype=torch.long, device=self.device)

                if isinstance(self.decoder, model.BinaryClassifierNet):
                    target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                                 dtype=torch.float, device=self.device)
                    loss = self.trainClassifier(input_tensor, target_tensor)
                else:
                    target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                                 dtype=torch.long, device=self.device)
                    loss = self.trainSeq(input_tensor, target_tensor)

            print_loss_total += loss
            loss_total += loss

            if iter % print_every == 0 and iter != 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                             iter * self.batch_size, iter / n_iters * 100, print_loss_avg))

        return loss_total / n_iters

    # helper function for testing
    def testtree(self, X_list, Y_list, output_programs, vocab_path, vul_info):
        with torch.no_grad():
            input_amount = len(X_list) - (len(X_list) % self.batch_size)

            output_df = pd.DataFrame(columns=['CWE ID', 'CVE ID', 'Vulnerability Classification', 'predicted repair', 'actual repair', 'BLEU score', 'Match'])

            for evi in tqdm(range(0, input_amount, self.batch_size), desc="Evaluation : "):
                input_tree = X_list[evi:evi + self.batch_size]
                batched_tree = batch_tree_input(input_tree)
                target_tensor = torch.tensor(Y_list[evi:evi + self.batch_size], dtype=torch.long, device=self.device)

                # Initialize zero tensor for encoder outputs, but this will only be used if decoder attention is supported with tree encoder
                encoder_outputs = torch.zeros(self.batch_size, self.max_length, self.encoder.hidden_size,
                                              device=self.device)

                # returns tuple with hidden state and cell state
                encoder_state = self.encoder(batched_tree['features'], batched_tree['node_order'],
                                             batched_tree['adjacency_list'], batched_tree['edge_order'])

                hidden = torch.zeros(self.decoder.num_layers, self.batch_size, self.encoder.hidden_size, device=self.device)
                cell = torch.zeros(self.decoder.num_layers, self.batch_size, self.encoder.hidden_size, device=self.device)

                # Create index list for accessing all root node hidden and cell states
                tree_indices = batched_tree['tree_sizes']
                tree_indices.insert(0, 0)
                tree_indices.pop()

                for layer in range(self.decoder.num_layers):
                    j = 0
                    for index in tree_indices:
                        hidden[layer][j] = encoder_state[0][index]
                        cell[layer][j] = encoder_state[1][index]
                        j += 1

                decoder_hidden_cell = (hidden, cell)

                output_list = [[self.begin_token] for _ in range(self.batch_size)]
                decoder_input = torch.tensor([self.begin_token]*self.batch_size, device=self.device)

                for di in range(self.max_length):
                    decoder_output, decoder_hidden_cell = self.decoder(
                        decoder_input, decoder_hidden_cell, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    for ai in range(self.batch_size):
                        output_list[ai].append(decoder_input[ai].tolist())

                for i in range(self.batch_size):
                    translated_target = util.translate(target_tensor[i], "tensor", vocab_path, self.end_token)
                    translated_output = util.translate(output_list[i], "list", vocab_path, self.end_token)

                    bleu_score = metrics.calcBleu(translated_target, translated_output)
                    match = metrics.calcMatch(translated_target, translated_output)
                    output_df = output_df.append([{'CWE ID':vul_info.loc[evi+i, 'CWE ID'], 'CVE ID':vul_info.loc[evi+i, 'CVE ID'], 'Vulnerability Classification':vul_info.loc[evi+i, 'Vulnerability Classification'], 'predicted repair':translated_output, 'actual repair':translated_target, 'BLEU score':bleu_score, 'Match':match}], ignore_index=True)

            # clear cache on the GPU
            if torch.cuda.is_available():
                del encoder_state
                gc.collect()
                torch.cuda.empty_cache()

            self.eval(output_df, output_programs)

    def testseq(self, X_test, Y_test, output_programs, vocab_path, vul_info):
        with torch.no_grad():
            input_amount = len(X_test) - (len(X_test) % self.batch_size)

            output_df = pd.DataFrame(columns=['CWE ID', 'CVE ID', 'Vulnerability Classification', 'predicted repair', 'actual repair', 'BLEU score', 'Match'])

            for evi in tqdm(range(math.floor(input_amount / self.batch_size)), desc="Evaluation : "):
                input_tensor = torch.tensor(X_test[evi * self.batch_size:(evi + 1) * self.batch_size], dtype=torch.long,
                                            device=self.device)
                target_tensor = torch.tensor(Y_test[evi * self.batch_size:(evi + 1) * self.batch_size], dtype=torch.long,
                                             device=self.device)

                encoder_hidden_cell = self.encoder.initHidden()

                encoder_outputs, encoder_hidden_cell = self.encoder(input_tensor.view(self.batch_size, self.max_length), encoder_hidden_cell)

                # Reshape hidden cell tuple to concat bidirectional values
                hidden = encoder_hidden_cell[0].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
                cell = encoder_hidden_cell[1].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
                # Set initial decoder hidden and cell values with encoder values
                decoder_hidden_cell = (hidden, cell)

                decoder_input = torch.tensor([self.begin_token] * self.batch_size, dtype=torch.long, device=self.device)
                output_list = [[self.begin_token] for _ in range(self.batch_size)]

                for di in range(self.max_length):
                    decoder_output, decoder_hidden_cell = self.decoder(
                        decoder_input, decoder_hidden_cell, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    for i in range(self.batch_size):
                        output_list[i].append(decoder_input[i].tolist())


                for i in range(self.batch_size):
                    translated_target = util.translate(target_tensor[i], "tensor", vocab_path, self.end_token)
                    translated_output = util.translate(output_list[i], "list", vocab_path, self.end_token)

                    bleu_score = metrics.calcBleu(translated_target, translated_output)
                    match = metrics.calcMatch(translated_target, translated_output)
                    output_df = output_df.append([{'CWE ID': vul_info.loc[evi + i, 'CWE ID'],
                                                   'CVE ID': vul_info.loc[evi + i, 'CVE ID'],
                                                   'Vulnerability Classification': vul_info.loc[
                                                       evi + i, 'Vulnerability Classification'],
                                                   'predicted repair': translated_output,
                                                   'actual repair': translated_target, 'BLEU score': bleu_score,
                                                   'Match': match}], ignore_index=True)

            # clear cache on the GPU
            if torch.cuda.is_available():
                del encoder_outputs
                gc.collect()
                torch.cuda.empty_cache()

            self.eval(output_df, output_programs)

    def eval(self, output_df, output_programs):
        matchScores = sum(output_df['Match']) / len(output_df['Match'])

        print(f'The accuracy is : {matchScores}')

        meanScores = sum(output_df['BLEU score']) / len(output_df['BLEU score'])
        varScores = sum([((x - meanScores) ** 2) for x in output_df['BLEU score']]) / len(output_df['BLEU score'])
        stdScores = varScores ** 0.5

        print(f'The average bleu score is : {meanScores}')
        print(f'The std of the bleu score is : {stdScores}')

        if output_programs == "test":
            print('Saving best 10 programs to file...')
            util.save_best_10_programs(output_df)
            print('Print average BLEU score per CWE...')
            util.print_CWE_average(output_df)

    def testClas(self, X_test, Y_test, vul_info):
        with torch.no_grad():
            input_amount = len(X_test) - (len(X_test) % self.batch_size)

            match = 0
            total = 0

            CWE_accuracy_df = pd.DataFrame(columns=['CWE', 'match', 'total', 'accuracy'])

            for evi in tqdm(range(math.floor(input_amount / self.batch_size)), desc="Evaluation : "):
                labels = torch.tensor(Y_test[evi * self.batch_size:(evi + 1) * self.batch_size], dtype=torch.long,
                                             device=self.device)

                if isinstance(self.encoder, model.SeqEncoderLSTM):
                    input_tensor = torch.tensor(X_test[evi * self.batch_size:(evi + 1) * self.batch_size], dtype=torch.long,
                                                device=self.device)

                    encoder_hidden_cell = self.encoder.initHidden()

                    encoder_outputs, encoder_hidden_cell = self.encoder(input_tensor.view(self.batch_size, self.max_length), encoder_hidden_cell)

                    # Reshape hidden cell tuple to concat bidirectional values
                    hidden = encoder_hidden_cell[0].view(1, self.batch_size, self.encoder.hidden_size*self.encoder.num_layers)

                    # clear cache on the GPU
                    if torch.cuda.is_available():
                        del encoder_outputs
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    input_tree = X_test[evi * self.batch_size:(evi + 1) * self.batch_size]
                    tree_batch = batch_tree_input(input_tree)

                    # returns tuple with hidden state and cell state
                    encoder_state = self.encoder(tree_batch['features'], tree_batch['node_order'],
                                                 tree_batch['adjacency_list'],
                                                 tree_batch['edge_order'])

                    hidden = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)

                    # Create index list for accessing all root node hidden and cell states
                    tree_indices = tree_batch['tree_sizes']
                    tree_indices.insert(0, 0)
                    # Pop the last indice as it does not reference the position of a tree roots hidden state
                    tree_indices.pop()

                    j = 0
                    for index in tree_indices:
                        hidden[0][j] = encoder_state[0][index]
                        j += 1

                    # clear cache on the GPU
                    if torch.cuda.is_available():
                        del encoder_state
                        gc.collect()
                        torch.cuda.empty_cache()

                prediction = self.decoder(hidden).view(self.batch_size)

                for i in range(self.batch_size):
                    total += 1
                    if round(prediction[i].item()) == labels[i].item():
                        match += 1

                    cwe_id = vul_info.loc[evi + i, 'CWE ID']
                    CWE_accuracy_df = util.calc_clas_CWE_accuracy(CWE_accuracy_df, cwe_id, prediction[i], labels[i])

            print("Accuracy per CWE of model is :")
            print(CWE_accuracy_df.sort_values('accuracy', ascending=False))
            print(f"Total accuracy of model is {match/total} of {total} datapoints")
