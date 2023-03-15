import Config

from Model.CustomAlexnet import *

def get_model_by_name(name):
    if name == 'alexnet1':
        return CustomAlexnet(class_num=Config.class_num)
    elif name == 'alexnet2':
        return CustomAlexnet(class_num=Config.class_num)
