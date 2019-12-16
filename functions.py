import numpy as np
import cv2

def rgb2gray(rgb):
    '''
    take a numpy rgb image return a new single channel image converted to greyscale
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def process_image(obs, rows, cols):

    im = np.dot(obs[...,:3], [0.299, 0.587, 0.114])

    obs = rgb2gray(im)
    obs = cv2.resize(obs, (rows, cols))
    return obs

def image_to_ascii(im):
    asc = []
    chars = ["B","S","#","&","@","$","%","*","!",":","."]
    for j in range(im.shape[1]):
        line = []
        for i in range(im.shape[0]):
            line.append(chars[int(im[i, j]) // 25])
        asc.append("".join(line))

    for line in asc:
        print(line)