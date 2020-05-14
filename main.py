import argparse
import logging
import os
import time

from data import processPoetry,build_vocabulary,Vocabulary,Poetry
from model import LSTMPoetry
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
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
    parser.add_argument('--batch-size', default=15, help='The number of batch-size.', type=int)
    parser.add_argument('--embedding-dim', default=300, help='The dimension of embedding .', type=int)
    parser.add_argument('--hidden-dim', default=150, help='The dimension of hidden .', type=int)
    parser.add_argument('--hidden-dim', default=150, help='The dimension of hidden .', type=int)
    parser.add_argument('--num_layers', default=2, help='The rnn layers.', type=int)
    parser.add_argument('--bidirectional', action='store_true', help='Whether using bidirectional RNNs')
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
                       sents_len=args.sentnum,num_layers=args.num_layers)

if __name__ == '__main__':
    main()