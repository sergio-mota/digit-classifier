import torch


class CNN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            pool_kernel_size,
            pool_stride
        ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.pool = torch.nn.MaxPool2d(pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class FC(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            is_output
        ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=not is_output)

    def forward(self, x):
        x = self.linear(x)
        return x

class CNN_FC(torch.nn.Module):
    def __init__(
            self,
            input_size: list[int],
            num_conv_maps: list[int],
            conv_kernel_sizes: list[int],
            pool_kernel_sizes: list[int],
            pool_strides: list[int],
            fc_sizes: list[int]
        ):
        super().__init__()

        if not (
            len(num_conv_maps) == 
            len(conv_kernel_sizes) == 
            len(pool_kernel_sizes) == 
            len(pool_strides)
        ):
            raise ValueError("convolutional layers must have the same length")

        conv_layers = []
        for i in range(len(num_conv_maps)):
            conv_layers.append(CNN(
                num_conv_maps[i-1] if i > 0 else 1,
                num_conv_maps[i],
                conv_kernel_sizes[i],
                pool_kernel_sizes[i],
                pool_strides[i]
            ))

            input_size = [(x - conv_kernel_sizes[i] + 1) // pool_strides[i] for x in input_size]

        fc_layers = []
        for i in range(len(fc_sizes)):
            fc_layers.append(FC(
                fc_sizes[i-1] if i > 0 else num_conv_maps[-1] * input_size[0] * input_size[1],
                fc_sizes[i],
                i==(len(fc_sizes) - 1)
            ))
        
        self.all_layers = torch.nn.Sequential(
            *(conv_layers + [torch.nn.Flatten()] + fc_layers)
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits
