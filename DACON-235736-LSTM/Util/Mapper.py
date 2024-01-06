import Config

from Model.SimpleLSTM import *

def get_model_by_name(model_name):
    if model_name == 'simplelstm':
        return SimpleLSTM(input_size=Config.input_size, output_size=Config.output_size)
