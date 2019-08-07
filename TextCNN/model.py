from torch import nn
import torch

class TextCNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextCNN, self).__init__()
        self.config = config

        # Embedding 层，加载预训练好的词向量
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[0]),
            nn.ReLU(),
            # MaxPool1d 网上都是 max_sen_len - 卷积的 kernel_size + 1
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1)
        )    # output : (batch_size, num_channels, 1)  最后的 1 是经过了 maxpool
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size,
                      out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1)
        )

        # dropout
        self.dropout = nn.Dropout(self.config.dropout_keep)

        # 全连接层
        self.fc = nn.Linear(self.config.num_channels*len(self.config.kernel_size), self.config.output_size)

        # softmax 层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = (batch_size, max_sen_len)
        embedded_sent = self.embeddings(x).permute(1, 2, 0)  # embeddings 的输出为 (max_sen_len, batch_size, embed_size)
        # embedded_sent.shape = (batch_size=64, embed_size=300, max_sen_len = 30)

        # 分别经过三个不同 kernel_size 的卷积层
        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        # 将三个不同 kernel_size 的结果拼接
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)

        # dropout
        final_feature_map = self.dropout(all_out)

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
