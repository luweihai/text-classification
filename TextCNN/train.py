# train.py

from utils import *
from model import *
from config import Config
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    config = Config()
    train_file = "../data/ag_news.train"
    test_file = "../data/ag_news.test"
    w2v_file = 'E:/data/word_embedding/glove.840B.300d/glove.840B.300d.txt'

    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)



    # 创建 model、optimizer、loss function
    model = TextCNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()   # train()启用 BatchNormalization 和 Dropout，如果是 eval() 就不启用
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    CrossEntropyLoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(CrossEntropyLoss)

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = run_epoch(dataset.train_iterator, dataset.val_iterator, i, config, model)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    # 模型在测试数据上评估
    model.eval()
    test_acc = evaluate_model(model, dataset.test_iterator)
    print("Final Testing Accuracy: {:.4f}".format(test_acc))
