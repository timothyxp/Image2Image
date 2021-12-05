from torch import nn
from .unet_layers import ConvBlock, DownBlock, UpBlock
from i2i.datasets.collator import I2IBatch


class UNet(nn.Module):
    def __init__(
        self, feature_levels_num, input_ch_size, hidden_ch_size, output_ch_size, block_depth,
        kernel_size=3, max_ch_num=1e9, dropout_p=0, *args, **kwargs
    ):
        """
        Input:
            n_classes - number of classes
            feature_levels_num - number of down- and up- block levels
            input_ch_size - input number of channels (1 for gray images, 3 for rgb)
            hidden_ch_size - output number of channels of the first Convolutional Block (in the original paper - 32)
            block_depth - number of convolutions + activations in one Convolutional Block
            kernel_size - kernel size for all convolution layers
        """
        super(UNet, self).__init__()
        self.input_block = ConvBlock(input_ch_size, hidden_ch_size, 1, block_depth)
        self.down_blocks = []
        self.up_blocks = []
        self.feature_levels_num = feature_levels_num
        self.dropout_p = dropout_p

        cur_ch_num = hidden_ch_size
        for _ in range(feature_levels_num):
            output_ch_num = min(cur_ch_num * 2, max_ch_num)

            self.down_blocks.append(DownBlock(cur_ch_num, output_ch_num, kernel_size, block_depth, self.dropout_p))
            self.up_blocks.append(UpBlock(output_ch_num, cur_ch_num, kernel_size, block_depth, self.dropout_p))
            cur_ch_num *= 2
            cur_ch_num = min(cur_ch_num, max_ch_num)

        self.center_conv = ConvBlock(cur_ch_num, cur_ch_num, kernel_size, block_depth)

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(list(reversed(self.up_blocks)))
        self.output_block = ConvBlock(hidden_ch_size, output_ch_size, 1, block_depth)

    def forward(self, batch: I2IBatch):
        x = self.input_block(batch.sketch_images)

        outputs = []
        for i in range(self.feature_levels_num):
            mem, x = self.down_blocks[i](x)

            outputs.append(mem)

        x = self.center_conv(x)

        outputs = outputs[::-1]

        for i in range(self.feature_levels_num):
            x = self.up_blocks[i](outputs[i], x)

        batch.predicted_image = self.output_block(x)

        return batch
