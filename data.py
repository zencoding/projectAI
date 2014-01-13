import gzip, cPickle

def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	data = cPickle.load(f)
	f.close()
	return data
