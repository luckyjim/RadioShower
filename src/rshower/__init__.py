"""
Mega Hertz radio
"""
__version__ = "1.0.0"
__author__= "Colley Jean-Marc"

#PATH_MODEL = "/home/jcolley/projet/grand_wk/data/model"
PATH_MODEL = ""


def set_path_model_du(s_path):
    global PATH_MODEL
    PATH_MODEL = s_path


def get_path_model_du():
    return PATH_MODEL
