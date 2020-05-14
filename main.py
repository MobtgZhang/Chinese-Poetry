import argparse
import logging
import os
import time
from tqdm import tqdm
import random
import torch
from torch.autograd import Variable

from data import processPoetry,build_vocabulary,Vocabulary,Poetry
from model import LSTMPoetry
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def train(model,dataset,criterion,optimizer,args,device):
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        pin_memory=args.cuda,
    )
    model.train()
    optimizer.zero_grad()
    for epoch in range(args.epoches):
        total_loss = 0.0
        for item in tqdm(data_loader, desc="Training epoch " + str(epoch + 1) + ''):
            (in_title, in_poetry), (tar_title, tar_poetry) = item
            in_title,in_poetry = Variable(in_title).to(device),Variable(in_poetry).to(device)
            tar_title, tar_poetry = Variable(tar_title).to(device),Variable(tar_poetry).to(device)
            titoutput, output = model.forward(in_title,in_poetry)
            loss1 = criterion.forward(titoutput.transpose(1, 2), tar_title)
            Length = len(output)
            loss2 = criterion.forward(output.transpose(1, 2), tar_poetry.reshape(Length, -1))
            loss = loss1+loss2
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print("loss is {},epoch is {}".format(total_loss/len(data_loader),epoch+1))
    torch.save(model,)
def main():
    parser = argparse.ArgumentParser(
        description='PyTorch RNNs for Poetry Generation')
    # data arguments
    parser.add_argument('--datadir', default='data', help='path to dataset',type=str)
    parser.add_argument('--rawdir', default=None, help='path to raw dataset',type=str)
    parser.add_argument('--logdir', default='log', help='path to log', type=str)
    parser.add_argument('--tag',default='tang',help='poetry type for the project.',type=str)
    parser.add_argument('--wordnum',default=5,help='The number of poetry words in the sentences.',type=int)
    parser.add_argument('--sentnum', default=4, help='The number of poetry sentences.', type=int)
    parser.add_argument('--max-len', default=20, help='The number of poetry titles.', type=int)
    parser.add_argument('--embedding-dim', default=300, help='The dimension of embedding .', type=int)
    parser.add_argument('--hidden-dim', default=150, help='The dimension of hidden .', type=int)
    parser.add_argument('--num_layers', default=2, help='The rnn layers.', type=int)
    parser.add_argument('--batch-size', default=30, help='The batch-size of the dataset.', type=int)
    parser.add_argument('--data-workers', type=int, default=5,help='Number of subprocesses for data loading')
    parser.add_argument('--epoches', default=50, help='The batch-size of the dataset.', type=int)
    parser.add_argument('--bidirectional', action='store_true', help='Whether using bidirectional RNNs')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--seed', default=123, type=int, help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    # preparing log
    # logging defination
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_name = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    log_dir = os.path.join(os.getcwd(), args.logdir, )
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, model_name + ".log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(args)
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    assert (args.rawdir is not None)
    # preparing dataset
    poetry_path = os.path.join(args.datadir, "poet.%s._%d_%d.json" % (args.tag, args.sentnum, args.wordnum))
    if os.path.exists(poetry_path):
        logger.info("The poetry dataset has been built in path: %s" % poetry_path)
    else:
        logger.info("Preparing poetry...")
        processPoetry(args.rawdir,args.datadir,sentNum=args.sentnum,wordsNum=args.wordnum,
                      max_title_len=args.max_len,tag=args.tag)
        logger.info("Poetry processed!")
    # preparing vocabulary
    vocab_path = os.path.join(args.datadir, "vocab.txt")
    if os.path.exists(vocab_path):
        logger.info("The vocabulary has been built in path: %s" % os.path.join(args.datadir, "vocab.txt"))
    else:
        logger.info("Building vocabulary...")
        build_vocabulary(args.rawdir, args.datadir)
        logger.info("The vocabulary has been built.")
    VocabDataSet = Vocabulary(vocab_path)

    PoetryDataSet = Poetry(VocabDataSet,args.max_len,poetry_path)

    # preparing model

    model = LSTMPoetry(vocab_size=len(VocabDataSet),embedding_dim=args.embedding_dim,hidden_dim=args.hidden_dim,
                       sents_len=args.sentnum,num_layers=args.num_layers,name = model_name)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device), criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # training process
    logger.info("Begin training model!")
    train(model,PoetryDataSet,criterion,optimizer,args,device)
    logger.info("End training model!")
if __name__ == '__main__':
    main()