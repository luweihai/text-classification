# model.py

from torch import nn
import torch
import torch.nn.functional as F

class Seq2SeqAttention(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(Seq2SeqAttention, self).__init__()
        self.config = config

        # Embedding 层，加载预训练好的词向量
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        #    self.embeddings = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        # LSTM 层作为 Encode
        self.lstm = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers,
                            bidirectional=self.config.bidirectional
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

    def apply_attention(self, rnn_output, final_hidden_state):

        '''

        Apply Attention on RNN output

        我们将注意计算每个隐藏状态和 LSTM 的最后一个隐藏状态之间对应的软对齐分数。我们将使用 torch.bmm 进行批量矩阵乘法

        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN

        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch

        '''

        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)  # (batch_size, seq_len)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)  # (max_sen_len, batch_size, embed_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)

        # 获取最后一层的 h_n
        batch_size = h_n.shape[1]  # 注意下面 view 的时候最好不要用固定的 batch_size, 不然有时候不够 batch_size 的话就会报错
        h_n_final_layer = h_n.contiguous().view(
            self.config.hidden_layers,
            self.config.bidirectional + 1,
            batch_size,
            self.config.hidden_size
        )[-1,:,:,:]   # 取最后一层的 h_n，这里区别与普通的 LSTM

        # Attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)
# 不能这样，这里拼接不能用 view，不然数据的顺序会变       final_hidden_state = h_n_final_layer.contiguous().view((self.config.bidirectional + 1) * batch_size, self.config.hidden_size)

        rnn_output = lstm_out.permute(1, 0, 2)
        attention_out = self.apply_attention(rnn_output, final_hidden_state)
        # attention_out: (batch_size, num_directions * hidden_size)
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)

        # dropout
        final_feature_map = self.dropout(concatenated_vector)  # (num_layers * num_directions, batch_size, hidden_size)

        # 全连接
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)  # 返回 softmax 的结果

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):  # 学习率衰减
        print("Reducing learning rate: ")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
