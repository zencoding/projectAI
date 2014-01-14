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
