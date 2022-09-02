import math
from black import out
import torch
import torch.nn as nn
from ops import reparameterize


class ConvolutionalBlock(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ConvolutionalBlock, self).__init__()

        midc = int(outc * scale)
        self.eps = 1e-4

        if inc is not outc:
            self.conv_expand = nn.Conv2d(
                in_channels=inc,
                out_channels=outc,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            )
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(
            in_channels=inc,
            out_channels=midc,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(midc, eps=self.eps)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=midc,
            out_channels=outc,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(outc, eps=self.eps)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.relu2(self.bn2(self.conv2(output)))
        return output


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    "if inc is not outc" -> "if inc != outc"
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(
                in_channels=inc,
                out_channels=outc,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            )
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(
            in_channels=inc,
            out_channels=midc,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=midc,
            out_channels=outc,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Conv2dBatchNorm(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding=0, groups=1):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.eps = 1e-4
        self.batch_norm = nn.BatchNorm2d(out_size, eps=self.eps)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class InceptionResnetBlock(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(InceptionResnetBlock, self).__init__()

        self.eps = 1e-4
        midc = int(outc * scale)
        assert outc % 2 == 0

        if inc is not outc:
            self.conv_expand = nn.Conv2d(
                in_channels=inc,
                out_channels=outc,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            )
        else:
            self.conv_expand = None

        self.branch_0 = Conv2dBatchNorm(
            inc, outc // 2, kernel_size=1, stride=1, groups=groups
        )
        self.branch_1 = nn.Sequential(
            Conv2dBatchNorm(inc, midc, kernel_size=1, stride=1, groups=groups),
            Conv2dBatchNorm(midc, outc // 2, kernel_size=1, stride=1, groups=groups),
        )
        self.conv = nn.Conv2d(outc, outc, kernel_size=1, stride=1, groups=groups)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        output = torch.cat((x0, x1), dim=1)
        output = self.conv(output)
        return self.relu(torch.add(output, identity_data))


def get_conv_class(arch):
    if arch == "conv":
        return ConvolutionalBlock
    elif arch == "res":
        return ResidualBlock
    elif arch == "inception":
        return InceptionResnetBlock
    else:
        raise ValueError()


class Encoder(nn.Module):
    def __init__(
        self,
        arch="res",
        cdim=3,
        zdim=512,
        channels=(64, 128, 256, 512, 512, 512),
        image_size=256,
    ):
        super(Encoder, self).__init__()
        self.conv_block = get_conv_class(arch)

        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc, eps=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module(
                "res_in_{}".format(sz),
                self.conv_block(cc, ch, scale=1.0),
            )
            self.main.add_module("down_to_{}".format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module("res_in_{}".format(sz), self.conv_block(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        arch="res",
        cdim=3,
        zdim=512,
        channels=(64, 128, 256, 512, 512, 512),
        image_size=256,
        conv_input_size=None,
    ):
        super(Decoder, self).__init__()
        self.conv_block = get_conv_class(arch)

        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.LeakyReLU(
                0.2, inplace=True
            ),  # limit output before convolutions to avoid NANs when using amp
        )

        sz = int(math.sqrt(num_fc_features // cc))

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module(
                "res_in_{}".format(sz),
                self.conv_block(cc, ch, scale=1.0),
            )
            self.main.add_module(
                "up_to_{}".format(sz * 2), nn.Upsample(scale_factor=2, mode="nearest")
            )
            cc, sz = ch, sz * 2

        self.main.add_module("res_in_{}".format(sz), self.conv_block(cc, cc, scale=1.0))
        self.main.add_module("predict", nn.Conv2d(cc, cdim, 5, 1, 2))
        self.main.add_module("sigmoid", nn.Sigmoid())

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class SoftIntroVAE(nn.Module):
    def __init__(
        self,
        arch="res",
        cdim=3,
        zdim=512,
        channels=(64, 128, 256, 512, 512, 512),
        image_size=256,
    ):
        super(SoftIntroVAE, self).__init__()

        self.zdim: int = zdim
        self.cdim: int = cdim

        self.encoder = Encoder(
            arch,
            cdim,
            zdim,
            channels,
            image_size,
        )

        self.decoder = Decoder(
            arch,
            cdim,
            zdim,
            channels,
            image_size,
            conv_input_size=self.encoder.conv_output_size,
        )

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z):
        y = self.decode(z)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu")):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y
