import torch
import random
import time
import util
import metrics
import model
import math
from treelstm import batch_tree_input
from tqdm import tqdm


class Trainer(object):
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, max_length,
                 teacher_forcing_ratio, device, begin_token, end_token):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.begin_token = begin_token
        self.end_token = end_token

    # helper function for training
    def trainTree(self, input_tree, target_tensor):
        # Store target sequence length, minus 1 since start token does not need to be predicted
        target_length = target_tensor.size(1) - 1

        # Initialize zero tensor for encoder outputs, but this will only be used if decoder attention is supported with tree encoder
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, self.encoder.hidden_size, device=self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = 0

        # returns tuple with hidden state and cell state
        encoder_state = self.encoder(input_tree['features'], input_tree['node_order'], input_tree['adjacency_list'], input_tree['edge_order'])

        hidden = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)
        cell = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)

        # Create index list for accessing all root node hidden and cell states
        tree_indices = input_tree['tree_sizes']
        tree_indices.insert(0,0)
        tree_indices.pop()

        j = 0
        for index in tree_indices:
            hidden[0][j] = encoder_state[0][index]
            cell[0][j] = encoder_state[1][index]
            j+=1

        decoder_hidden_cell = (hidden, cell)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # print("Training with teacher forcing...")
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    target_tensor[:, di].view(self.batch_size, 1), decoder_hidden_cell, encoder_outputs)

                # calculate the loss separate for every output in the batch
                loss += self.criterion(decoder_output, target_tensor[:, di + 1])
        else:
            # print("Training with no teacher forcing...")
            # Initialize sequence with start token
            decoder_input = torch.tensor([self.begin_token]*self.batch_size, dtype=torch.long, device=self.device)

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    decoder_input.view(self.batch_size, 1), decoder_hidden_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        loss.backward()

        # clear cache and variables to free space on the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def trainSeq(self, input_tensor, target_tensor):
        encoder_hidden_cell = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1) - 1

        encoder_outputs = torch.zeros(self.batch_size, self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden_cell = self.encoder(
                input_tensor[:, ei].view(self.batch_size, 1), encoder_hidden_cell)
            encoder_outputs[:, ei] = encoder_output[0, :]

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
                    target_tensor[:, di].view(self.batch_size, 1), decoder_hidden_cell, encoder_outputs)

                # calculate the loss separate for every output in the batch
                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        else:
            # print("Training with no teacher forcing...")
            # Initialize sequence with start token
            decoder_input = torch.tensor([self.begin_token] * self.batch_size, dtype=torch.long, device=self.device)

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden_cell = self.decoder(
                    decoder_input.view(self.batch_size, 1), decoder_hidden_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di + 1])

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / (target_length)

    def trainIters(self, X_list, Y_list, n_iters, print_every=100):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0

        for iter in range(n_iters):
            if isinstance(self.encoder, model.ChildSumTreeLSTM):
                input_tree = X_list[iter * self.batch_size:(iter + 1) * self.batch_size]
                batched_trees = batch_tree_input(input_tree)
                #target_tensor = torch.tensor(Y_list[iter], dtype=torch.long, device=self.device)
                target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                             dtype=torch.long, device=self.device)

                loss = self.trainTree(batched_trees, target_tensor)
            else:
                input_tensor = torch.tensor(X_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                            dtype=torch.long, device=self.device)
                target_tensor = torch.tensor(Y_list[iter * self.batch_size:(iter + 1) * self.batch_size],
                                             dtype=torch.long, device=self.device)

                loss = self.trainSeq(input_tensor, target_tensor)

            print_loss_total += loss
            epoch_loss_total += loss

            if iter % print_every == 0 and iter != 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                             iter * self.batch_size, iter / n_iters * 100, print_loss_avg))

        epoch_loss_avg = print_loss_total / n_iters
        print('Average loss for epoch is %.4f (%s)' % (epoch_loss_avg, util.timeSince(start, n_iters)))


    # helper function for testing
    def testtree(self, X_list, Y_list):
        with torch.no_grad():
            input_amount = len(X_list) - (len(X_list) % self.batch_size)

            bleuScores = []
            matches = []

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

                hidden = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)
                cell = torch.zeros(1, self.batch_size, self.encoder.hidden_size, device=self.device)

                # Create index list for accessing all root node hidden and cell states
                tree_indices = batched_tree['tree_sizes']
                tree_indices.insert(0, 0)
                tree_indices.pop()

                j = 0
                for index in tree_indices:
                    hidden[0][j] = encoder_state[0][index]
                    cell[0][j] = encoder_state[1][index]
                    j += 1

                decoder_hidden_cell = (hidden, cell)

                output_tensor = [[self.begin_token] for _ in range(self.batch_size)]
                decoder_input = torch.tensor([self.begin_token]*self.batch_size, device=self.device)

                for di in range(self.max_length):
                    decoder_output, decoder_hidden_cell = self.decoder(
                        decoder_input, decoder_hidden_cell, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    for ai in range(self.batch_size):
                        output_tensor[ai].append(decoder_input[ai].tolist())

                for ci in range(self.batch_size):
                    bleuScores.append(metrics.calcBleu(target_tensor[ci], output_tensor[ci], self.end_token))
                    matches.append(metrics.calcMatch(target_tensor[ci], output_tensor[ci]))

            self.eval(bleuScores, matches)

    def testseq(self, X_test, Y_test):
        with torch.no_grad():
            input_amount = len(X_test)
            input_length = 1000

            bleuScores = []
            matches = []

            for i in tqdm(range(math.floor(input_amount / self.batch_size)), desc="Evaluation : "):
                input_tensor = torch.tensor(X_test[i * self.batch_size:(i + 1) * self.batch_size], dtype=torch.long,
                                            device=self.device)
                target_tensor = torch.tensor(Y_test[i * self.batch_size:(i + 1) * self.batch_size], dtype=torch.long,
                                             device=self.device)

                # Initialize encoder output sequence with zero
                encoder_outputs = torch.zeros(self.batch_size, self.max_length, self.encoder.hidden_size,
                                              device=self.device)
                encoder_hidden_cell = self.encoder.initHidden()

                for ei in range(input_length):
                    encoder_output, encoder_hidden_cell = self.encoder(
                        input_tensor[:, ei].view(self.batch_size, 1), encoder_hidden_cell)
                    encoder_outputs[:, ei] = encoder_output[0, :]

                # Reshape hidden cell tuple to concat bidirectional values
                hidden = encoder_hidden_cell[0].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
                cell = encoder_hidden_cell[1].view(self.encoder.num_layers, self.batch_size, self.encoder.hidden_size)
                # Set initial decoder hidden and cell values with encoder values
                decoder_hidden_cell = (hidden, cell)

                decoder_input = torch.tensor([self.begin_token] * self.batch_size, dtype=torch.long, device=self.device)
                output_tensor = [[self.begin_token] for _ in range(self.batch_size)]

                for di in range(self.max_length):
                    decoder_output, decoder_hidden_cell = self.decoder(
                        decoder_input, decoder_hidden_cell, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    for i in range(self.batch_size):
                        output_tensor[i].append(decoder_input[i].tolist())

                for i in range(self.batch_size):
                    bleuScores.append(metrics.calcBleu(target_tensor[i], output_tensor[i], self.end_token))
                    matches.append(metrics.calcMatch(target_tensor[i], output_tensor[i]))

            self.eval(bleuScores, matches)

    def eval(self, bleuScores, matches):
        matchScores = sum(matches) / len(matches)

        print(f'The accuracy is : {matchScores}')

        meanScores = sum(bleuScores) / len(bleuScores)
        varScores = sum([((x - meanScores) ** 2) for x in bleuScores]) / len(bleuScores)
        stdScores = varScores ** 0.5

        print(f'The average bleu score is : {meanScores}')
        print(f'The std of the bleu score is : {stdScores}')