import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
random_seed = 1234
torch.manual_seed(random_seed)


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1, decoder_bias=False):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            dim_feedforward=embed_dim,  # 不要使用默认的2048，太大了
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.decoder_bias = decoder_bias
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        if decoder_bias:
            self.fc_weight = nn.Linear(input_dim, output_dim+1)
        else:
            self.fc_weight = nn.Linear(input_dim, output_dim)
            self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        # self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        # torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        # 事实上 x的形状为[seq_len, input_dim]，没有batch维度
        # 里面的weight_tokens加入的方式也很奇怪，明明根本没有用到它，直接丢掉吧
        pos_wave = x
        # x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        if self.decoder_bias:
            transformer_output = self.transformer_encoder(x)
            # weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
            weights_bias = self.fc_weight(transformer_output + pos_wave)
            weights = weights_bias[:, :-1]
            bias = weights_bias[:, -1]  # seq_len
        else:
            x = torch.cat([x, self.bias_token], dim=0)
            transformer_output = self.transformer_encoder(x)
            # weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
            weights = self.fc_weight(transformer_output[: -1] + pos_wave)

            bias = self.fc_bias(
                transformer_output[-1]
        )  # Using the last output to generate bias
        # ZSJ weight输出形状为seq_len, output_dim, bias输出形状为1, embed_dim
        return weights, bias



class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out



class Dynamic_MLP_OFA_Encoder(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, embed_dim=1024):
        super().__init__()
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.embed_dim
        self.num_patches = -1 

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, metadata):
        # wvs的形状就是M，也就是通道数量，他没有batch_size维度，意味着这个batch里面各个样本的模态是一样的，一般来说确实
        # ZSJ 此处有个encoder的强假设，inplanes=波长数量=输入特征的通道数=M
        # 因此encoder过程先把wvs变成形状为M,D的tensor，M为波长数量，D为波长通道数，
        # 然后把它输入到Transformer Encoder里面，得到形状为M,K*K*embeddim的weight和1,embeddim的bias
        # 然后就可以把weight变成（M,embeddim，K,K)的卷积，bias变成（1，embeddim）的偏置
        # 但是这个假设在decoder中需要调整，因为波长数量等于输出特征通道数了
        # 因为需要形状为（embeddim, M, K,K)的weight和（1，M）的bias
        # 让Transformer Encoder里面的embeddim+1可以解决这个问题

        # 这个类对linear encoder使用


        # img_feat shape: B,L,C; metadata shape: B,C
        metadata = metadata[0]  # C
        inplanes = metadata.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        metadata = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, metadata* 1000)

        # print(waves.shape)
        metadata = self.fclayer(metadata)
        # print(waves.shape)
        weight, bias = self._get_weights(metadata)  # 3x3x3
        # print(weight.shape)
        # print(bias.shape)


        dynamic_weight = weight.view(
            self.embed_dim, inplanes
        )  # 3xoutdx16x16  很奇怪，这里不应该使用view操作，而应该使用transpose操作，view会改变值的连续分布顺序，但是偏偏这样做是没有问题的，太奇怪了，以实验为准吧

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        # )
        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=1, padding=1, dilation=1
        # )
        dynamic_out = F.linear(img_feat, weights, bias=bias)

        x = dynamic_out
        # x = x.flatten(2).transpose(1, 2)

        # return x, waves
        return x  # B,L,D



class Dynamic_MLP_OFA_Decoder(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """
    # 这个类对linear decoder使用

    def __init__(self, wv_planes, embed_dim=1024, decoder_bias=False):
        super().__init__()
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = self.embed_dim
        self.num_patches = -1
        self.decoder_bias = decoder_bias
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim, decoder_bias=decoder_bias
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, metadata):
        # img_feat shape: B,D,H,W; metadata shape: B,C

        metadata = metadata[0]  # C
        inplanes = metadata.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        metadata = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, metadata* 1000)

        # print(waves.shape)
        metadata = self.fclayer(metadata)
        # print(waves.shape)
        weight, bias = self._get_weights(metadata)  # 3x3x3
        # print(weight.shape)
        # print(bias.shape)

        dynamic_weight = weight.view(
            inplanes, self.embed_dim
        )  # 3xoutdx16x16
        if bias is not None and self.decoder_bias:
            bias = bias.view([inplanes]) * self.scaler

        weights = dynamic_weight * self.scaler

        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        # )
        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=None, stride=1, padding=1, dilation=1
        # )
        if self.decoder_bias:
            # print(f'bias: {bias}')
            # print(f'weights: {weights}')
            dynamic_out = F.linear(img_feat, weights, bias=bias)
        else:
            dynamic_out = F.linear(img_feat, weights)

        x = dynamic_out
        # x = x.flatten(2).transpose(1, 2)

        # return x, waves
        # return x, weights
        return x



class Dynamic_Conv_OFA_Patch_Embed(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wv_planes = wv_planes
        embed_dim = wv_planes
        self.embed_dim = wv_planes
        if type(kernel_size) == list or type(kernel_size) == tuple:
            self._num_kernel = self.kernel_size[0] * self.kernel_size[1] * self.embed_dim
        else:
            self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, metadata):
        # wvs的形状就是M，也就是通道数量，他没有batch_size维度，意味着这个batch里面各个样本的模态是一样的，一般来说确实
        # ZSJ 此处有个encoder的强假设，inplanes=波长数量=输入特征的通道数=M
        # 因此encoder过程先把wvs变成形状为M,D的tensor，M为波长数量，D为波长通道数，
        # 然后把它输入到Transformer Encoder里面，得到形状为M,K*K*embeddim的weight和1,embeddim的bias
        # 然后就可以把weight变成（M,embeddim，K,K)的卷积，bias变成（1，embeddim）的偏置
        # 但是这个假设在decoder中需要调整，因为波长数量等于输出特征通道数了
        # 因为需要形状为（embeddim, M, K,K)的weight和（1，M）的bias
        # 让Transformer Encoder里面的embeddim+1可以解决这个问题

        # 这个类对卷积使用

        # img_feat shape: B,C,H,W; metadata shape: B,C
        metadata = metadata[0]  # C
        inplanes = metadata.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        metadata = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, metadata)

        # print(waves.shape)
        metadata = self.fclayer(metadata)
        # print(waves.shape)
        weight, bias = self._get_weights(metadata)  # 3x3x3
        # print(weight.shape)
        # print(bias.shape)
        if type(self.kernel_size) == list or type(self.kernel_size) == tuple:
            dynamic_weight = weight.view(
                self.embed_dim, inplanes, self.kernel_size[0], self.kernel_size[1]
            )  # 3xoutdx16x16
        else:
            dynamic_weight = weight.view(
                self.embed_dim, inplanes, self.kernel_size, self.kernel_size
            )  # 3xoutdx16x16
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        # )
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.stride, padding=self.padding
        )

        x = dynamic_out
        # x = x.flatten(2).transpose(1, 2)

        # return x, waves
        return x  # B,D,H,W
    

class Dynamic_MLP_OFA_Projection(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """
    # 这个类对反卷积使用

    def __init__(self, wv_planes, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        embed_dim = wv_planes
        self.embed_dim = wv_planes
        self.stride = stride
        self.padding = padding
        if type(kernel_size) == list or type(kernel_size) == tuple:
            self._num_kernel = self.kernel_size[0] * self.kernel_size[1] * self.embed_dim
            self.patch_size = kernel_size
        else:
            self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
            self.patch_size = (kernel_size, kernel_size)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, metadata):
        # img_feat shape: B,D,H,W; metadata shape: B,C

        metadata = metadata[0]  # C
        inplanes = metadata.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        metadata = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, metadata)

        # print(waves.shape)
        metadata = self.fclayer(metadata)
        # print(waves.shape)
        weight, bias = self._get_weights(metadata)  # 3x3x3
        # print(weight.shape)
        # print(bias.shape)

        if type(self.kernel_size) == list or type(self.kernel_size) == tuple:
            dynamic_weight = weight.view(
                self.embed_dim, inplanes, self.kernel_size[0], self.kernel_size[1]
            )  # 3xoutdx16x16
        else:
            dynamic_weight = weight.view(
                self.embed_dim, inplanes, self.kernel_size, self.kernel_size
            )  # 3xoutdx16x16
        # if bias is not None:
        #     bias = bias.view([inplanes]) * self.scaler

        weights = dynamic_weight * self.scaler

        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        # )
        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=None, stride=1, padding=1, dilation=1
        # )
        dynamic_out = F.conv_transpose2d(
            img_feat, weights, bias=None, stride=self.stride, padding=self.padding
        )
        x = dynamic_out
        # x = x.flatten(2).transpose(1, 2)

        # return x, waves
        # return x, weights
        return x