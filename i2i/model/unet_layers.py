from torch import nn


class ConvBlock(nn.Module):
    '''
       Convolutional Block, includes several sequential convolutional and activation layers.
       Hint: include BatchNorm here
    '''

    def __init__(self, input_ch, output_ch, kernel_size, block_depth):
        '''
            input_ch - input channels num
            output_ch - output channels num
            kernel_size - kernel size for convolution layers
            block_depth - number of convolution + activation repetitions
        '''
        super().__init__()

        conv_list = []
        for i in range(block_depth):
            conv_list.append(nn.Conv2d(
                input_ch, output_ch, (kernel_size, kernel_size),
                padding=(kernel_size - 1) // 2)
            )

            conv_list.append(nn.BatchNorm2d(output_ch))
            conv_list.append(nn.ReLU())
            input_ch = output_ch

        self.conv_net = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.conv_net(x)
        return x


class DownBlock(nn.Module):
    '''
        Encoding block, includes pooling (for shape reduction) and Convolutional Block (ConvBlock)
    '''

    def __init__(self, input_ch, output_ch, kernel_size, block_depth):
        super().__init__()

        self.conv_block = ConvBlock(input_ch, output_ch, kernel_size, block_depth)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        mem = self.conv_block(x)
        # print(mem.shape, "downBlock")

        x = self.pool(mem)
        # print(x.shape, "downBlock")
        return mem, x


class UpBlock(nn.Module):
    '''
        Decoding block, includes upsampling and Convolutional Block (ConvBlock)
    '''

    def __init__(self, input_ch, output_ch, kernel_size, block_depth):
        super().__init__()

        self.conv = nn.ConvTranspose2d(input_ch, input_ch,
                                       2, stride=2, padding=0, output_padding=0)
        self.block = ConvBlock(input_ch, output_ch, kernel_size, block_depth)

    def forward(self, copied_input, lower_input):
        '''
            copied_input - feature map from one of the encoder layers
            lower_input - feature map from previous decoder layer
        '''

        lower_input = self.conv(lower_input, output_size=copied_input.size())
        # print(lower_input.shape, 'upBlock')
        x = lower_input + copied_input
        # print(x.shape, "upBlock")
        x = self.block(x)
        # print(x.shape, "upBlock")

        return x
