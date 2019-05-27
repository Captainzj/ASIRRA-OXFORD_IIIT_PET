import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init

import math
from OXFORD_IIIT.cliqueNet_pytorch.utils import transition, global_pool, compress, clique_block

class build_cliquenet(nn.Module):
    def __init__(self, input_channels, list_channels, list_layer_num, if_att):
        super(build_cliquenet, self).__init__()
        self.fir_trans = nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.fir_bn = nn.BatchNorm2d(input_channels)
        self.fir_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_num = len(list_channels)

        self.if_att = if_att
        self.list_block = nn.ModuleList()
        self.list_trans = nn.ModuleList()
        self.list_gb = nn.ModuleList()
        self.list_gb_channel = []
        self.list_compress = nn.ModuleList()
        input_size_init = 56

        for i in range(self.block_num):

            if i == 0:
                self.list_block.append(clique_block(input_channels=input_channels, channels_per_layer=list_channels[0], layer_num=list_layer_num[0], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(input_channels + list_channels[0] * list_layer_num[0])  #将gb(全局池化层)当做block中下一个layer, 输入所有前置层特征
            else :
                self.list_block.append(clique_block(input_channels=list_channels[i-1] * list_layer_num[i-1], channels_per_layer=list_channels[i], layer_num=list_layer_num[i], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])

            if i < self.block_num - 1:  # 非最后一个block
                self.list_trans.append(transition(self.if_att, current_size=input_size_init, input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8))

            # 输出为1*1的特征图 # 先compress后gb # compress: 使得特征通道减半
            self.list_gb.append(global_pool(input_size=input_size_init, input_channels=self.list_gb_channel[i] // 2))
            self.list_compress.append(compress(input_channels=self.list_gb_channel[i], keep_prob=0.8))  # bn-1*1 conv
            input_size_init = input_size_init // 2  # 过渡层 list_trans[i] 最后是一个pooling层 使得特征图大小减半

        self.fc = nn.Linear(in_features=sum(self.list_gb_channel) // 2, out_features=37)  # compress: 使得特征通道减半

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)
        output = self.fir_bn(output)
        output = F.relu(output)
        output = self.fir_pool(output)

        feature_I_list = []

        # use stage II + stage II mode
        for i in range(self.block_num):
            block_feature_I, block_feature_II = self.list_block[i](output)
            block_feature_I = self.list_compress[i](block_feature_I)  # block_feature_I -> compress
            feature_I_list.append(self.list_gb[i](block_feature_I))   # after_compress_block_feature_I -> global-pool -> 1*1 特征图
            if i < self.block_num - 1:
                output = self.list_trans[i](block_feature_II)   # update output (after transition layer) for next Clique_block

        final_feature = feature_I_list[0]  # final_feature of block-one
        for block_id in range(1, len(feature_I_list)):
            final_feature = torch.cat((final_feature, feature_I_list[block_id]), 1)   # concatenate the output of block

        final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        output = self.fc(final_feature)

        return output
