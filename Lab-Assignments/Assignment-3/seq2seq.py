from __future__ import unicode_literals, print_function, division

import re
import time
import string
import random
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import math
from io import open
plt.switch_backend('agg')

import warnings
warnings.filterwarnings("ignore")


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_TOKEN = 0 
EOS_TOKEN = 1
MAX_LENGTH = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def tensorFromSentence(lang, sentence):
    indices = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]
    return torch.tensor(indices, dtype = torch.long, device = DEVICE).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor  = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    
    return (input_tensor, target_tensor)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru       = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).unsqueeze(0)

        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = DEVICE)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru       = nn.GRU(hidden_size. hidden_size)
        self.output    = nn.Linear(hidden_size, output_size)
        self.softmax   = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.output(output[0]))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = DEVICE)

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length = MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p   = dropout_p
        self.max_length  = max_length

        self.embedding    = nn.Embedding(self.output_size, self.hidden_size)
        self.attention    = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout      = nn.Dropout(self.dropout_p)
        self.gru          = nn.GRU(self.hidden_size, self.hidden_size)
        self.output       = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        concatenation     = torch.cat((embedded[0], hidden[0]), dim = 1)
        attention_outputs = self.attention(concatenation)
        attention_weights = F.softmax(attention_outputs, dim = 1)

        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attention_applied[0]), dim = 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.output(output[0]), dim = 1)

        return output, hidden, attention_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = DEVICE)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length    =  input_tensor.size(0)
    target_length   = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = DEVICE)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input  = torch.tensor([[SOS_TOKEN]], device = DEVICE)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing: 
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def train_loop(encoder, decoder, pairs, steps, print_freq = 5000, plot_freq = 1000, learning_rate = 0.01):
    start = time.time()
    plot_losses = []

    print_loss_total = 0
    plot_loss_total  = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(steps)]

    for step in range(1, steps + 1):
        batch = training_pairs[step - 1]
        input_tensor, target_tensor = batch[0], batch[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss 
        plot_loss_total  += loss 

        if step % print_freq == 0:
            print_loss_avg   = print_loss_total / print_freq
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, step / steps), step, step / steps * 100, print_loss_avg))

        if step % plot_freq == 0:
            plot_loss_avg = plot_loss_total / plot_freq
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device = DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n = 30):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def readLangs(path_to_data, lang1, lang2):
    lines = open(path_to_data, encoding = 'utf-8') \
        .read().strip().split('\n')

    pairs = []
    for line in lines:
        pairs.append((normalizeString(line.split('\t')[0]), normalizeString(line.split('\t')[1])))

    print("Read %s sentence pairs" % len(pairs))
    pairs = [pair for pair in pairs if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH]
    print("Trimmed to %s sentence pairs" % len(pairs))

    input_lang  = Lang(lang1)
    output_lang = Lang(lang2)
    
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

if __name__ == "__main__":
    PATH_TO_DATA = 'data/ron-eng/ron.txt'
    input_lang, output_lang, pairs = readLangs(PATH_TO_DATA, 'eng', 'ro')

    print(random.choice(pairs))

    hidden_size = 512
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
    decoder = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p = 0.1).to(DEVICE)
    
    train_loop(encoder, decoder, pairs, steps = 150000)

    evaluateRandomly(encoder, decoder)

