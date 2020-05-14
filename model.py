# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSentence(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_layers=2,bidirectional = False):
        super(LSTMSentence, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,bidirectional = bidirectional,batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        if hidden is None:
            h_0,c_0 = self.init_hid(batch_size)
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        hid = self.linear(output)
        pred = self.softmax(hid)
        return pred,hidden
    def predict(self,words:torch.LongTensor):
        assert words.dim()==1
        h_0,c_0 = self.init_hid(1)
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(words)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        hid = self.linear(output)
        pred = torch.argmax(self.softmax(hid),dim=2)
        return pred
    def init_hid(self,batch_size):
        num_directions = 2 if self.bidirectional else 1
        h_0 = input.data.new(num_directions * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        c_0 = input.data.new(num_directions * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        return h_0,c_0
#表示七言绝句诗词还是五言绝句诗词还是七言律诗还是五言律诗
#        律诗一般是8句，绝句是4句，标题的长度没有限制
class LSTMPoetry(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim,sents_len=8,num_layers=2,bidirectional = False):
        super(LSTMPoetry, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sents_len = sents_len
        self.layers = []
        for k in range(sents_len):
            self.layers.append(LSTMSentence(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional))
        self.title_rnn = nn.LSTM(hidden_dim,hidden_dim,num_layers=num_layers,bidirectional=bidirectional)
        self.title_linear = nn.Linear(hidden_dim,vocab_size,bias=False)
    def forward(self,title,sents):
        batch_size =title.size(0)
        output = []
        h,c = self.layers[0].init_hid(batch_size)#(num_directions*num_layers) batch * vocab_size
        out_title = title
        for k in range(self.sents_len-1):
            out,(h,c) = self.layers[k].forward(sents[k+1],(h,c)) #batch * seq_len * vocab_size
            out_title,_ = self.title_rnn(out_title,(h,c))
            out_title = F.gelu(out_title)
            output.append(out)
        # 首尾呼应
        out, (h, c) = self.layers[self.sents_len-1].forward(sents[0], (h, c))  # batch * seq_len * vocab_size
        out_title, _ = self.title_rnn(out_title, (h, c))
        out_title = F.gelu(out_title)
        output.append(out)

        output = torch.cat(output,dim=1)
        self.title_linear(out_title)
        out_title = F.softmax(out_title,dim=2)
        return out_title,output
