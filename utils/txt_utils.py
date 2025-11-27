import yaml
from easydict import EasyDict

def read_from_yaml(txt_path):
    with open(txt_path, "r", encoding="utf-8") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return EasyDict(y)
