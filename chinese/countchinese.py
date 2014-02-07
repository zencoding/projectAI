import gzip,cPickle
import numpy as np

"""Script to create list of 50 classes and then obtain all characters of these classes"""

classes = 50
list_of_classes = []

#Note load_chinese needs a folder with many pickles, as created by the readchinese.py script

def load_chinese(file_id):
    f = gzip.open('dataset/chinese'+str(file_id)+'pkl.gz','rb')
    x_train, t_train = cPickle.load(f)
    f.close()
    return x_train,t_train

i = 0
while len(list_of_classes) < classes:
    if t_train[i] not in list_of_classes:
        list_of_classes.append(t_train[i])
    i += 1

print list_of_classes
    
g = gzip.GzipFile('chineseclasslist.pkl.gz', 'wb')
cPickle.dump(list_of_classes,g,-1)
g.close()

print "success!"
filtered_x_train = []
filtered_t_train = []
for file_id in xrange(1,90):
    x_train, t_train = load_chinese(file_id)
    for i in xrange(len(t_train)):
        if t_train[i] in list_of_classes:
            filtered_x_train.append(x_train[:,i])
            filtered_t_train.append(t_train[i])


filtered_x_train = np.array(filtered_x_train)
print filtered_x_train.shape
g = gzip.GzipFile('chinesefiltered.pkl.gz', 'wb')
cPickle.dump((filtered_x_train,filtered_t_train),g,-1)
g.close()



        



