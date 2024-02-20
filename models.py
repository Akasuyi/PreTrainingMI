import torch
import torch.nn as nn
from torch.nn import init
import math

from torch.autograd import Variable

class ShallowNet_modified(torch.nn.Module):
    """
    modified shallowNet, similar conv layer but with other layers like GRU and self-attention.
    """

    def __init__(
            self,
            in_chans,
            n_classes,
            input_window_samples=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=150,
            pool_time_stride=15,
            final_conv_length=30,
            pool_mode="mean",
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
    ):
        super(ShallowNet_modified, self).__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        self.conv_time_m1_m2 = nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1), stride=1)
        self.conv_spat_m2 = nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                                    (1, self.in_chans), stride=1, bias=not self.batch_norm)

        self.norm = nn.BatchNorm2d(self.n_filters_spat, momentum=self.batch_norm_alpha, affine=True)

        self.pooling = torch.nn.AvgPool2d(kernel_size=(self.pool_time_length, 1),
                                          stride=(self.pool_time_stride, 1))

        self.dropout = nn.Dropout(p=self.drop_prob)

        num_head = 1
        # self.attention = torch.nn.MultiheadAttention(n_filters_spat, num_head)

        self.linear = torch.nn.Linear(2560, self.n_classes)

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time_m1_m2.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time_m1_m2.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat_m2.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat_m2.bias, 0)
        if self.batch_norm:
            init.constant_(self.norm.weight, 1)
            init.constant_(self.norm.bias, 0)
        init.xavier_uniform_(self.linear.weight, gain=1)
        init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # k represent constant number
        # x has been transformed into pytorch tensors
        # x: [batch_size, num_channel, num_time_points]

        # x -> [batch_size, 1, num_channel, num_time_points]
        x = x.unsqueeze(1)
        # x -> [batch_size, 1, num_time_points, num_channel]
        x = x.transpose(2, 3)

        # x -> [batch_size, num_time_filter, num_time_points-k, num_channel]
        x = self.conv_time_m1_m2(x)

        # x -> [batch_size, num_spect_filter, num_time_points-k, 1]
        x = self.conv_spat_m2(x)
        x = self.norm(x)

        x = x * x
        # x -> [batch_size, num_spect_filter, (num_time_points-k)//pool_kernel_size, 1]
        x = self.pooling(x)

        x = self.safe_log(x)
        x = self.dropout(x)
        
        # x -> [batch_size, (((num_time_points-k)//pool_kernel_size) - k2) * num_spect_filter]
        x = x.reshape(x.size(0), -1)

        return x
    
    def classify(self, x):
        x = self(x)
        x = self.linear(x)
        return x
    
    def safe_log(self, x, eps=1e-6):
        """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=eps))


class deep_net_modified(nn.Module):
    """Deep ConvNet model from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
    in_chans : int
        XXX

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_pool_mode="max",
        later_pool_mode="max",
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_pool_mode = first_pool_mode
        self.later_pool_mode = later_pool_mode
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        
        self.conv_time_m1_m2 = nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                )
        self.conv_spat_m2 = nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                )
        n_filters_conv = self.n_filters_spat

        self.bnorm = nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1))
        self.conv_pool_block = nn.ModuleList()
        
        for num_filters, filter_length in zip([[n_filters_conv, self.n_filters_2], [self.n_filters_2, self.n_filters_3], [self.n_filters_3, self.n_filters_4]], [self.filter_length_2, self.filter_length_3, self.filter_length_4]):
            self.conv_pool_block.append(nn.Dropout(p=self.drop_prob))
            self.conv_pool_block.append(nn.Conv2d(
                    num_filters[0],
                    num_filters[1],
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ))
            self.conv_pool_block.append(nn.BatchNorm2d(
                        num_filters[1],
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ))
            self.conv_pool_block.append(nn.ELU())
            self.conv_pool_block.append(nn.MaxPool2d(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ))

        self.linear = nn.Linear(1800, self.n_classes)

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time_m1_m2.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time_m1_m2.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat_m2.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat_m2.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        for i, param in enumerate(self.conv_pool_block):
            if (i+1) % 5 == 2:
                init.xavier_uniform_(param.weight, gain=1)
                if not self.batch_norm:
                    init.constant_(param.bias, 0)
            if (i+1) % 5 == 3:
                init.constant_(param.weight, 1)
                init.constant_(param.bias, 0)

        init.xavier_uniform_(self.linear.weight, gain=1)
        init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.permute(0, 3, 2, 1)
        x = self.conv_time_m1_m2(x)
        x = self.conv_spat_m2(x)
        x = self.bnorm(x)
        x = self.elu(x)
        x = self.pool(x)
        for block in self.conv_pool_block:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        # x = self.linear(x)
        return x
    
    def classify(self, x):
        x = self(x)
        x = self.linear(x)
        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet_MI(torch.nn.Module):
    def __init__(self, num_channel, num_class, F1, F2, D, dropout_rate):
        super(EEGNet_MI, self).__init__()
        kernel_size_1 = 64
        self.conv2d_m1_m2 = torch.nn.Conv2d(1, F1, kernel_size=(1, kernel_size_1),
                                      bias=False, stride=1, padding=(0, kernel_size_1 // 2))
        self.batch_norm = torch.nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)
        self.deepwise_conv2d_m2 = Conv2dWithConstraint(
                F1,
                F1 * D,
                (num_channel, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=F1,
                padding=(0, 0),
            )
        self.batch_norm_2 = torch.nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3)
        self.elu = torch.nn.ELU()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.deepwise_conv2d_2 = torch.nn.Conv2d(D * F1, F2, kernel_size=(1, 16),
                                                 bias=False, groups=D * F1, padding=(0, 16 // 2))
        self.elemwise_conv2d = torch.nn.Conv2d(F2, F2, kernel_size=(1, 1),
                                               bias=False, padding=(0, 0))
        self.batch_norm_3 = torch.nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.avg_pool_2 = torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.linear = torch.nn.Linear(560, num_class)

        self.softmax = torch.nn.LogSoftmax(dim=1)
        _glorot_weight_zero_bias(self)

    def forward(self, x):
        # x has been transformed into pytorch tensors
        # x: [batch_size, num_channel, num_time_points]

        # x -> [batch_size, 1, num_channel, num_time_points]
        x = x.unsqueeze(1)
        # x -> [batch_size, F1, num_channel, num_time_points]
        x = self.conv2d_m1_m2(x)

        # x -> [batch_size, F1, num_channel, num_time_points]
        x = self.batch_norm(x)

        # x -> [batch_size, F1*D, 1, num_time_points]
        x = self.deepwise_conv2d_m2(x)
        

        # normalize and activate x without changing x's dimension
        x = self.batch_norm_2(x)
        x = torch.nn.functional.elu(x)

        # average pooling 2D
        # x -> [batch_size, F1*D, 1, num_time_points//4]
        x = self.avg_pool(x)

        # dropout
        x = self.dropout(x)

        # Separable Convolution = deepwise+elementwise convolution
        # x -> [batch_size, F2, 1, num_time_points//4]
        x = self.elemwise_conv2d(self.deepwise_conv2d_2(x))
        x = self.batch_norm_3(x)
        x = torch.nn.functional.elu(x)

        # average pooling
        # x -> [batch_size, F2, 1, num_time_points//32]
        x = self.avg_pool_2(x)
        x = self.dropout(x)

        # x -> [batch_size, F2 * num_time_points//32]
        x = x.reshape(x.size(0), -1)

        return x
    
    def classify(self, x):
        x = self(x)
        result = self.linear(x)
        return result


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class FineTuneNet(nn.Module):
    def __init__(self, backbone, test_classes, size_after_backbone):
        super(FineTuneNet, self).__init__()
        self.backbone = backbone
        self.test_head = torch.nn.Linear(size_after_backbone, test_classes)

    def forward(self, x):
        backbone_result = self.backbone(x)
        fine_tune_result = self.test_head(backbone_result)
        return fine_tune_result

    def re_initialize(self):
        init.xavier_uniform_(self.test_head.weight, gain=1)
        init.constant_(self.test_head.bias, 0)
