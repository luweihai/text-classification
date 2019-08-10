# model.py

from torch import nn
import torch

class TextRNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextRNN, self).__init__()
        self.config = config

        # Embedding 层，加载预训练好的词向量
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        #    self.embeddings = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        # LSTM 层
        self.lstm = nn.LSTM(input_size= self.config.embed_size,
                            hidden_size= self.config.hidden_size,
                            num_layers= self.config.hidden_layers,
                            dropout= self.config.dropout_keep,
                            bidirectional= self.config.bidirectional
                            )
        '''
            输入LSTM中的X数据格式尺寸为(seq_len, batch, input_size)，此外h0和c0尺寸如下
            h0(num_layers * num_directions,  batch_size,  hidden_size)
            c0(num_layers * num_directions,  batch_size,  hidden_size)
            
            LSTM输出的数据格式尺寸为(seq_len, batch, hidden_size * num_directions)；输出的hn和cn尺寸如下
            hn(num_layers * num_directions,  batch_size,  hidden_size)
            cn(num_layers * num_directions,  batch_size,  hidden_size)
        '''

        # dropout
        self.dropout = nn.Dropout(self.config.dropout_keep)

        # 全连接层
        self.fc = nn.Linear(  # 就是 hn、cn 的输出然后去掉 batch_size
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size
        )

        # softmax 层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)  # (max_sen_len = 30, batch_size=64, embed_size=300)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)

        # dropout
        final_feature_map = self.dropout(h_n) # (num_layers * num_directions, batch_size, hidden_size)

        # 变成 (batch_size, hidden_size * hidden_layers * num_directions)
        final_feature_map = final_feature_map.permute(1, 0, 2)  # (batch_size, hidden_layers * num_directions, hidden_size)
        final_feature_map = final_feature_map.contiguous().view(final_feature_map.size()[0], final_feature_map.size()[1] * final_feature_map.size()[2])
        # 全连接
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)  # 返回 softmax 的结果

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):     # 学习率衰减
        print("Reducing learning rate: ")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
