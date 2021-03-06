import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        # <YOUR CODE HERE>
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        
    def forward(self, hidden, encoder_outputs):
        # <YOUR CODE HERE>
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # hidden = hidden.unsqueeze(1)
        # print(hidden.shape)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attention = torch.sum(self.attn(torch.cat([hidden, encoder_outputs], dim = 2)), dim=2)
        
        return F.softmax(attention, dim=1)
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)#, bidirectional=True) # <YOUR CODE HERE>
        
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # <YOUR CODE HERE>
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
      
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat([embedded, weighted], dim = 2)
        # output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output, hidden = self.rnn(rnn_input)#, hidden)
        
        # embedded = embedded.squeeze(0)
        # output = output.squeeze(0)
        # weighted = weighted.squeeze(0)
        # [output, weighted, embedded]
        prediction = self.out(torch.cat([embedded, hidden, weighted], dim = 2))
        
        return prediction.squeeze(0), hidden
        
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
