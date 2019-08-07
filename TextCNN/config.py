# config.py

class Config(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 4   # 一共四种类别
    max_epochs = 10
    lr = 0.1
    batch_size = 64
    max_sen_len = 30   # 截断
    dropout_keep = 0.8
