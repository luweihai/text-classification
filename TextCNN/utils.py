# utils.py

import torch
from torchtext.vocab import Vectors
from torchtext import data
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import spacy

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    def parse_label(self, label):
        '''
            label: __label__2 这个形式
            return: 2
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
            加载数据，转换为 dataframe 类型
        '''
        with open(filename, 'r') as datafile:
            data = [line.strip().split(',', maxsplit=1) for line in datafile]  #从左到右，只切分一次，避免句子也被切
            data_text = list(map(lambda x: x[1], data))   # 文本
            data_label = list(map(lambda x: self.parse_label(x[0]), data))  # label
        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df

    def load_data(self, w2v_file, train_file, test_file, val_file = None):
        '''
            从文件中读取数据，建立 iterators、vocabulary 和 embeddings
            Inputs:
                w2v_file(String): 预训练的词向量文件(Glove/Word2Vec)
                train_file(String): 训练数据路径
                test_file(String): 测试数据路径
                val_file(String): 验证数据路径
        '''

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # 创建 Field 对象
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        # LABEL 中的 sequential 一定要设置为 False
        LABEL = data.Field(sequential=False, use_vocab=False)  # 如果LABEL是整型，不需要 numericalize ， 就需要将 use_vocab=False
        datafields = [("text", TEXT), ("label", LABEL)]

        # 将 DataFrame 中的数据添加到 torchtext.data.Dataset 中
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]  # 生成训练样本
        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()] # 生成测试样本        text_data = data.Dataset(test_examples, datafields
        test_data = data.Dataset(test_examples, datafields)


        # 划分验证集
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_example = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_example, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8) # 利用 split 划分

        # 加载预训练的 word embedding
        TEXT.build_vocab(train_data, vectors= Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        # 生成训练数据迭代对象
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True
        )

        # 生成测试数据和验证数据的迭代对象
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False
        )
        print("Local {} train examples".format(len(train_data)))
        print("Local {} test examples".format(len(test_data)))
        print("Local {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for index, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score


def run_epoch(train_iterator, val_iterator, epoch, config, model):
    train_losses = []
    val_accuracies = []
    losses = []

    # 随着 epoch 的增加而动态衰减学习率
    if (epoch == int(config.max_epochs / 3)) or (epoch == int(2 * config.max_epochs / 3)):
        model.reduce_lr()

    for i, batch in enumerate(train_iterator):
        model.optimizer.zero_grad()
        if torch.cuda.is_available():
            x = batch.text.cuda()
            y = (batch.label - 1).type(torch.cuda.LongTensor)
        else:
            x = batch.text
            y = (batch.label - 1).type(torch.LongTensor)
        y_pred = model(x)
        loss = model.loss_op(y_pred, y)
        loss.backward()
        losses.append(loss.data.cpu().numpy())
        model.optimizer.step()

        if i % 100 == 0:
            print("Iter: {}".format(i + 1))
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)
            print("\tAverage training loss: {:.5f}".format(avg_train_loss))
            losses = []

            # Evalute Accuracy on validation set
            val_accuracy = evaluate_model(model, val_iterator)
            print("\tVal Accuracy: {:.4f}".format(val_accuracy))
            model.train()

    return train_losses, val_accuracies
