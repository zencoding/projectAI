# -*- coding: utf-8 -*- 
import math
import struct
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import gzip,cPickle

"""Script to preprocess the Chinese character dataset, needs folder hwdb1.1train with all the gnt files from the CASIA website"""

pickle_size = 10000
size = 40
samplenumber = 0
x_train = np.zeros((size*size,pickle_size))
t_train = [None] * pickle_size

for j in xrange(240):
    idfile = 1001+j
    filename = 'hwdb1.1train/'+str(idfile)+'-c.gnt'
    with open(filename,'rb') as f:
        while True:
            if (samplenumber != 0 and samplenumber % pickle_size == 0) or samplenumber == 1121749:
                print "Saving!"
                g = gzip.GzipFile('output/chinese'+str(samplenumber)+'.pkl.gz', 'wb')
                cPickle.dump((x_train,t_train),g,-1)
                g.close()
                x_train = np.zeros((size*size,pickle_size))
                t_train = [None] * pickle_size
            buf = f.read(4)
            if not buf: break
            samplesize = struct.unpack('i',buf)[0]
            char =  f.read(2).decode('gb2312')
            t_train[samplenumber % pickle_size] = char
            width = struct.unpack('H',f.read(2))[0]
            height = struct.unpack('H',f.read(2))[0]
            bitmap = struct.unpack('B'*width*height,f.read(width*height))
            # for binary
            # character = np.asarray(bitmap)
            character = np.asarray(bitmap)/255*255
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
            vector = 1-(np.array(background.getdata(), dtype=np.uint8).astype(np.float)/255)
            x_train[:,samplenumber % pickle_size] = vector
            print samplenumber
            samplenumber += 1

