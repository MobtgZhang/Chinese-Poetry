# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class LSTMSentence(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_layers=2,bidirectional = False):
        super(LSTMSentence, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,bidirectional = bidirectional,batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self, embput, hidden=None):
        batch_size = embput.size(0)
        if hidden is None:
            h_0,c_0 = self.init_hid(embput,batch_size)
        else:
            h_0, c_0 = hidden
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embput, (h_0, c_0))
        hid = self.linear(output)
        pred = self.softmax(hid)
        return pred,hidden
    def init_hid(self,input,batch_size):
        num_directions = 2 if self.bidirectional else 1
        h_0 = input.data.new(num_directions * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        c_0 = input.data.new(num_directions * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        return h_0,c_0
#表示七言绝句诗词还是五言绝句诗词还是七言律诗还是五言律诗
#        律诗一般是8句，绝句是4句，标题的长度没有限制
class LSTMPoetry(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim,sents_len=8,num_layers=2,bidirectional = False,name = None):
        super(LSTMPoetry, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sents_len = sents_len
        self.layers = []
        self.name = name
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        for k in range(sents_len):
            fc = LSTMSentence(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional)
            setattr(self, "fc%d" % k, fc)
            self.layers.append(fc)
        self.title_rnn = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,bidirectional=bidirectional,batch_first=True)
        self.title_linear = nn.Linear(hidden_dim*sents_len,vocab_size,bias=False)
    def forward(self,title,sents):
        '''
        :param title:(batch_size*seq_len)
        :param sents:(batch_size*sent_nums*seq_len)
        :return:
        '''
        batch_size,sent_len,seq_len = sents.size()
        assert (sent_len == self.sents_len)
        output = []
        h,c = self.layers[0].init_hid(sents,batch_size) #(num_directions*num_layers) batch * vocab_size
        titoutput = []
        for k in range(sent_len):
            embput = self.embeddings(sents[:,k,:])
            out,(h,c) = self.layers[k].forward(embput,(h,c)) #batch * seq_len * vocab_size
            titemb = self.embeddings(title)
            out_title,_ = self.title_rnn(titemb,(h,c))
            out_title = F.gelu(out_title)
            output.append(out)
            titoutput.append(out_title)
        output = torch.cat(output,dim=1)
        titoutput = torch.cat(titoutput,dim = 2)
        titoutput = self.title_linear(titoutput)
        titoutput = F.log_softmax(titoutput,dim=2)
        return titoutput,output
    def predict(self,words:set,max_len:int,wordnum:int):
        '''
        由一些输入的词语进行描写诗词中的语境来刻画诗词意境
        生成的方法如下：
        1.设输入的词语集合为Set,初始化h,c.随机从Set中抽取若干个词语(可以重复)组成生成第一句话的引导语句pre_sent1
        2.先生成第一句话pre_sent1,(h,c)->sent1,(h,c),随机抽取sent1中的若干个词语组成题目输入向量值，
        将(h,c)输入到题目生成器中，得到题目输出向量tit_v1
        3.将sent1中的词语加入到Set当中,随机从Set中抽取若干个词语(可以重复)组成生成第一句话的引导语句pre_sent2
        4.生成第二句话pre_sent2,(h,c)->sent2,(h,c),随机抽取sent2中的若干个词语组成题目输入向量值，
        将(h,c)输入到题目生成器中，得到题目输出向量tit_v2
        ...
        n.生成第n句话,pre_sent(n-1),(h,c)->sentn,(h,c),随机抽取sentn中的若干个词语组成题目输入向量值，
        将(h,c)输入到题目生成器中，得到题目输出向量tit_vn
        n+1.合并(tit_v1,tit_v2,...,tit_vn)，并输入到线性变换空间中，然后生成题目:
        注意在每一次输入题目预生词中词语的个数不超过题目的长度，其中的空位置用<BLK>标注，结尾用<EOS>标注。
        每一次输入的词语开头处必须是用<SOS>标注。
        :param words: 输入预生词的集合Set
        :param max_len:输入的题目长度
        :return: 输出诗句index
        '''
        words = list(words)
        pre_sent = []
        for k in range(wordnum):
            pre_sent.append(words[np.random.randint(0, len(words))])
        pre_sent = torch.LongTensor(pre_sent).unsqueeze(0)
        h, c = self.layers[0].init_hid(pre_sent,1)
        poetry = []
        for st_num in range(len(self.layers)):
            embput = self.embeddings(pre_sent)
            out, (h, c) = self.layers[st_num].forward(embput, (h, c))
            out = torch.argmax(out.squeeze(),dim=0)
            poetry.append(out.numpy().tolist())
            # 生成题目
            words = list(words)
            pre_sent = []
            for k in range(wordnum):
                pre_sent.append(words[np.random.randint(0,len(words))])
