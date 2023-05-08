import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def vietOCR_prediction(input):
    config = Cfg.load_config_from_name('vgg_transformer')
    #config['weights'] = './data/transformerocr.pth'
    config['device'] = 'cpu'
    config['cnn']['pretrained'] = True

    detector = Predictor(config)
    return detector.predict(input)
