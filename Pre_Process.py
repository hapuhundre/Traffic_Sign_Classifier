import numpy as np
import cv2
import matplotlib.pyplot as plt

def Norm(train,valid,test):
    t = (train - 128) / 128
    v = (valid - 128) / 128
    te = (test - 128) / 128
    return t,v,te
