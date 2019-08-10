# config.py

class Config(object):
    N = 1    # 有多少层的 Encoder
    d_model = 300   #  embedding_size
    d_ff = 600    # 2048 in Transformer Paper
    h = 10         # 分为几个 head
    dropout = 0.1
    output_size = 4
    lr = 0.0003
    max_epochs = 35
    batch_size = 128
    max_sen_len = 60