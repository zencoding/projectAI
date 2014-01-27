import gzip, cPickle

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def load_cifar():
    f = open('cifar100/train', 'rb')
    dictionary = cPickle.load(f)
    f.close()
    data = dictionary['data']
    return data


def load_ff():
    f = open('freyfaces.pkl','rb')
    data = cPickle.load(f)
    f.close()
    return data

def load_chinese(file_id):
    f = gzip.open('output4040/chinese'+str(file_id)+'0000.pkl.gz','rb')
    data = cPickle.load(f)
    f.close()
    return data


def load_filtered_chinese():
    f = gzip.open('chinesefiltered.pkl.gz','rb')
    data = cPickle.load(f)
    f.close()
