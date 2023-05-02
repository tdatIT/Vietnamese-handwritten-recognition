import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def vietOCR_prediction(path_file):
    config = Cfg.load_config_from_name('vgg_transformer')
    #https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA
    config['weights'] = './data/transformerocr.pth'
    config['device'] = 'cpu'
    config['cnn']['pretrained'] = True

    detector = Predictor(config)
    img = Image.open(path_file)

    return detector.predict(img)