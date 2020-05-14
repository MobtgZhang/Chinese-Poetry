import re
import os
import json
import torch
PAD_BLANK = 0
PAD_SOS = 1
PAD_EOS = 2
class Vocabulary:
    def __init__(self,filename = None):
        self.id2word = []
        self.word2id = {}
        self.add('<BLK>')
        self.add('<SOS>')
        self.add('<EOS>')
        if filename is not None:
            self.load(filename)
    def add(self,word):
        if word not in self.word2id:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
    def convertToidx(self,word:str):
        if word == ' ':
            return self.word2id['<BLK>']
        if word not in self.word2id:
            return None
        return self.word2id[word]
    def idxToword(self,idx:int):
        return self.id2word[idx]
    def load(self,filename):
        with open(filename,mode="r",encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                word = line.strip()
                self.add(word)
    def __len__(self):
        return len(self.id2word)

class Poetry(torch.utils.data.Dataset):
    def __init__(self,vocab:Vocabulary,max_len,filename:str = None):
        self.titles = []
        self.authors = []
        self.poetries = []
        self.max_len = max_len
        self.vocab = vocab
        if filename is not None:
            self.load(filename)
    def add_poetry(self,title,author,poetry):
        self.titles.append(title)
        self.authors.append(author)
        self.poetries.append(poetry)
    def load(self,filename):
        with open(filename,mode="r",encoding="utf-8") as f:
            outdata = json.load(f)
            for item in outdata:
                paragraphs = item['paragraphs']
                tmpParasIndexes = []
                for sent in paragraphs:
                    out = re.match("(.*)(，|？)(.*)(？|。)", sent)
                    paraIndexesA = self.readSentence(out.groups()[0].strip(),start='<SOS>',end='<EOS>')
                    paraIndexesB = self.readSentence(out.groups()[2].strip(),start='<SOS>',end='<EOS>')
                    tmpParasIndexes.append(paraIndexesA)
                    tmpParasIndexes.append(paraIndexesB)

                titleIndexes = self.readSentence(item['title'].strip(),MAX_LEN=self.max_len,start='<SOS>',end='<EOS>')
                self.titles.append(titleIndexes)
                self.authors.append(item['author'])
                self.poetries.append(tmpParasIndexes)
            self.poetries = torch.LongTensor(self.poetries)
            self.titles = torch.LongTensor(self.titles)
    def readSentence(self,sentence:str,MAX_LEN:int=None,start:str= None,end:str = None):

        if MAX_LEN is not None:
            sent_list = [PAD_BLANK]*MAX_LEN
            for index in range(len(sentence)):
                word = sentence[index]
                if self.vocab.convertToidx(word) is not None:
                    number = self.vocab.convertToidx(word)
                    sent_list[index] = number
        else:
            sent_list = []
            for word in sentence:
                if self.vocab.convertToidx(word) is not None:
                    index = self.vocab.convertToidx(word)
                    sent_list.append(index)
        if start is not None:
            sent_list = [self.vocab.convertToidx(start)] + sent_list
        if end is not None:
            sent_list.append(self.vocab.convertToidx(end))
        return sent_list
    def __getitem__(self, item):
        return (self.titles[item,:-1],self.poetries[item,:,:-1]),(self.titles[item,1:],self.poetries[item,:,1:])
    def __len__(self):
        return len(self.poetries)
def proessNumpoetry(paragraphs,sentNum,wordsNum):
    if len(paragraphs) == sentNum:
        # sentNum一般是2或者是4，这是对于七言律诗、五言律诗、七言绝句、五言绝句来说
        # 这里为了古诗的完整性，我将古诗的题目限制在20个字之内
        tmpsents = "".join(paragraphs)
        if len(tmpsents) == ((wordsNum+1)*sentNum*2) \
                and "□" not in tmpsents and (re.match("(.*)（(.*)）(.*)",tmpsents) is None):
            for sent in paragraphs:
                if len("".join(sent).split('，'))==2:
                    out = re.match("(.*)(，|？)(.*)(？|。)", sent)
                    if out is not None and (len(out.groups()[0])==wordsNum and len(out.groups()[2])==wordsNum):
                        return True
                    else:
                        break
                else:
                    break
    else:
        return False
def processPoetry(root_path:str,save_path:str,sentNum:int,wordsNum:int,max_title_len:int = 20,tag:str="tang"):
    pattern = "poet\.(%s)(.*)" % tag
    assert sentNum%2 == 0
    return_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for filename in os.listdir(root_path):
        if re.match(pattern,filename) is not None:
            file_path = os.path.join(root_path,filename)
            with open(file_path,mode="r",encoding="utf-8") as f:
                out = json.load(f)
                for item in out:
                    if proessNumpoetry(item['paragraphs'], sentNum//2,wordsNum) and (len(item['title']) <= max_title_len):
                        return_list.append(item)
    f_name = "poet.%s._%d_%d.json"%(tag,sentNum,wordsNum)
    save_filename = os.path.join(save_path,f_name)
    with open(save_filename,mode="w",encoding="utf-8") as f:
        json.dump(return_list,f)
def build_vocabulary(root_path,save_path):
    '''
    2E80-2EFF CJK 部首补充 2F00-2FDF 康熙字典部首
    3000-303F CJK 符号和标点 31C0-31EF CJK 笔画
    3200-32FF 封闭式 CJK 文字和月份 3300-33FF CJK 兼容
    3400-4DBF CJK 统一表意符号扩展 A 4DC0-4DFF 易经六十四卦符号
    4E00-9FBF CJK 统一表意符号 F900-FAFF CJK 兼容象形文字
    FE30-FE4F CJK 兼容形式 FF00-FFEF 全角ASCII、全角标点
    所以最后不包含以下的字符：
    0000-0127 ASCII码
    2E80-2EFF CJK 部首补充 2F00-2FDF 康熙字典部首
    3000-303F CJK 符号和标点 31C0-31EF CJK 笔画
    3200-32FF 封闭式 CJK 文字和月份 3300-33FF CJK 兼容
    4DC0-4DFF 易经六十四卦符号
    E000-F8FF
    FE30-FE4F CJK 兼容形式 FF00-FFEF 全角ASCII、全角标点
    '''
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pattern = "poet\.(tang|song)(.*)"
    exp_pattern = "[\u0000-\u0127]|[\u2E80-\u33FF]|[\u4DC0-\u4DFF]|[\uFE30-\uFFEF]|[\uE000-\uF8FF]"
    vocabulary = {}
    for filename in os.listdir(root_path):
        if re.match(pattern,filename) is not None:
            file_path = os.path.join(root_path,filename)
            with open(file_path,mode="r",encoding="utf-8") as f:
                out = json.load(f)
                for item in out:
                    lines = "".join(item['paragraphs'])
                    for word in lines:
                        if re.match(exp_pattern,word) is None:
                            if word not in vocabulary:
                                vocabulary[word] = len(vocabulary)+1
    with open(os.path.join(save_path,"vocab.txt"),mode="w",encoding="utf-8") as f:
        for word in vocabulary:
            f.write(word + "\n")
def CalculatePoetry(root_path:str,sentList:list,wordsNum:int,tag:str="tang"):
    '''
    仅仅做统计使用
    :param root_path:
    :param sentList:
    :param wordsNum:
    :param tag:
    :return:
    '''
    pattern = "poet\.(%s)(.*)" % tag
    for item in sentList:
        assert (item %2 == 0)
    numbers = {}
    for num in sentList:
        numbers[num] = 0
    for filename in os.listdir(root_path):
        if re.match(pattern, filename) is not None:
            file_path = os.path.join(root_path, filename)
            with open(file_path, mode="r", encoding="utf-8") as f:
                out = json.load(f)
                for item in out:
                    for num in sentList:
                        if proessNumpoetry(item['paragraphs'], num // 2, wordsNum):
                            numbers[num] += 1
                            continue
    return numbers