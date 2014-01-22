# -*- coding: utf-8 -*- 
import math
import struct
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

filename = '1001-c.gnt'
size = 60

with open(filename,'rb') as f:
    for i in xrange(5):
        samplesize = struct.unpack('i',f.read(4))[0]
        char =  f.read(2).decode('gb2312')
        width = struct.unpack('H',f.read(2))[0]
        height = struct.unpack('H',f.read(2))[0]
        bitmap = struct.unpack('B'*width*height,f.read(width*height))
        # for binary
        character = np.asarray(bitmap)/255*255
        # character = np.asarray(bitmap)
        character = character.reshape(height,width)
        im = Image.fromarray(np.uint8(character))
        im.thumbnail((size,size))
        background = Image.new('1', size = (size,size), color = (255))

        if width > height:
            offset = int(math.floor((size - ((float(size)/width) * height))/2))
            center = (0,offset)
        else:
            offset = int(math.floor((size - ((float(size)/height) * width))/2))
            center = (offset,0)

        background.paste(im,center)
        background.show()

