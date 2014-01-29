import gzip, cPickle

def load_mnist():
    f = gzip.open('dataset/mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def load_ff():
    f = open('dataset/freyfaces.pkl','rb')
    data = cPickle.load(f)
    f.close()
    return data


def load_filtered_chinese():
    f = gzip.open('dataset/chinesefiltered.pkl.gz','rb')
    data = cPickle.load(f)
    f.close()
    return data
