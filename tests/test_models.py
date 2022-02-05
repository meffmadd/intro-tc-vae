import sys
sys.path.append("..")

from models import SoftIntroVAE, ConvolutionalBlock, ResidualBlock, InceptionResnetBlock

def test_model_conv(mocker):
    conv_spy = mocker.spy(ConvolutionalBlock, "__init__")
    res_spy = mocker.spy(ResidualBlock, "__init__")
    inception_spy = mocker.spy(InceptionResnetBlock, "__init__")
    SoftIntroVAE(arch="conv")
    assert conv_spy.call_count > 0
    assert res_spy.call_count == 0
    assert inception_spy.call_count == 0


def test_model_res(mocker):
    conv_spy = mocker.spy(ConvolutionalBlock, "__init__")
    res_spy = mocker.spy(ResidualBlock, "__init__")
    inception_spy = mocker.spy(InceptionResnetBlock, "__init__")
    SoftIntroVAE(arch="res")
    assert conv_spy.call_count > 0
    assert res_spy.call_count > 0
    assert inception_spy.call_count == 0


def test_model_inception(mocker):
    conv_spy = mocker.spy(ConvolutionalBlock, "__init__")
    res_spy = mocker.spy(ResidualBlock, "__init__")
    inception_spy = mocker.spy(InceptionResnetBlock, "__init__")
    SoftIntroVAE(arch="inception")
    assert conv_spy.call_count == 0
    assert res_spy.call_count == 0
    assert inception_spy.call_count > 0
