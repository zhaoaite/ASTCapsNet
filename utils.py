import os
import scipy
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split


def load_casia(batch_size, is_training):
    data = np.load('data/casia/casia_gait_data.npy')
    label = np.loadtxt('data/casia/casia_gait_label.txt')
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((-1, 50, 50, 1)).astype(np.float32)
        trainY = trainY.reshape((15308)).astype(np.int32)	
        trX = trainX[:14000] / 10.
        trY = trainY[:14000]

        valX = trainX[14000:,] / 10.
        valY = trainY[14000:]

        num_tr_batch = 14000 // batch_size
        num_val_batch = 1308 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((-1, 50, 50, 1)).astype(np.float32)
        teY = teY.reshape((3827)).astype(np.int32)

        num_te_batch = 3827 // batch_size
        return teX / 10., teY, num_te_batch

    if is_training:
        return trX, trY
    else:
        return teX, teY



def load_oulp(batch_size, is_training):
    data = np.load('data/oulp/oulp_gait_data.npy')
    label = np.loadtxt('data/oulp/oulp_gait_label.txt')
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    trainX=trainX[:,5120:8120]
    teX=teX[:,5120:8120]
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((16320, 50, 60, 1)).astype(np.float32)
        trainY = trainY.reshape((16320)).astype(np.int32)	
        trX = trainX[:14688] / 10.
        trY = trainY[:14688]

        valX = trainX[14688:,] / 10.
        valY = trainY[14688:]

        num_tr_batch = 14688 // batch_size
        num_val_batch = 1632 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((4080, 50, 60, 1)).astype(np.float32)
        teY = teY.reshape((4080)).astype(np.int32)

        num_te_batch = 4080 // batch_size
        return teX / 10., teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY



def load_ga(batch_size, is_training):
    data = np.loadtxt('data/parkinson/Ga13592class4data.txt')
    label = np.loadtxt('data/parkinson/Ga13592class4label.txt')
    data=data[:,:950]
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((10873, 19, 50, 1)).astype(np.float32)
        trainY = trainY.reshape((10873)).astype(np.int32)	
        trX = trainX[:9000] / 100.
        trY = trainY[:9000]

        valX = trainX[9000:,] / 100.
        valY = trainY[9000:]

        num_tr_batch = 9000 // batch_size
        num_val_batch = 1873 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((2719, 19, 50, 1)).astype(np.float32)
        teY = teY.reshape((2719)).astype(np.int32)

        num_te_batch = 2719 // batch_size
        return teX / 100., teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY

def load_si(batch_size, is_training):
    data = np.loadtxt('data/parkinson/Si7744class4data.txt')
    label = np.loadtxt('data/parkinson/Si7744class4label.txt')
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((6195, 19, 100, 1)).astype(np.float32)
        trainY = trainY.reshape((6195)).astype(np.int32)	
        trX = trainX[:5500] / 100.
        trY = trainY[:5500]

        valX = trainX[5500:,] / 100.
        valY = trainY[5500:]

        num_tr_batch = 5500 // batch_size
        num_val_batch = 695 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((1549, 19, 100, 1)).astype(np.float32)
        teY = teY.reshape((1549)).astype(np.int32)

        num_te_batch = 1549 // batch_size
        return teX / 100., teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY



def load_ju(batch_size, is_training):
    data = np.loadtxt('data/parkinson/Ju11743class4data.txt')
    label = np.loadtxt('data/parkinson/Ju11743class4label.txt')

    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((9394, 19, 100, 1)).astype(np.float32)
        trainY = trainY.reshape((9394)).astype(np.int32)	
        trX = trainX[:8000] / 1000.
        trY = trainY[:8000]

        valX = trainX[8000:,] / 1000.
        valY = trainY[8000:]

        num_tr_batch = 8000 // batch_size
        num_val_batch = 1394 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((2349, 19, 100, 1)).astype(np.float32)
        teY = teY.reshape((2349)).astype(np.int32)

        num_te_batch = 2349 // batch_size
        return teX / 1000., teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY



def load_sdugait(batch_size, is_training):
    data = np.loadtxt('data/sdugait/12023f.txt')
    label = np.loadtxt('data/sdugait/12023label.txt')
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((9618, 7, 9, 1)).astype(np.float32)
        trainY = trainY.reshape((9618)).astype(np.int32)	
        trX = trainX[:8000]
        trY = trainY[:8000]

        valX = trainX[8000:,]
        valY = trainY[8000:]

        num_tr_batch = 8000 // batch_size
        num_val_batch = 1618 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((2405, 7, 9, 1)).astype(np.float32)
        teY = teY.reshape((2405)).astype(np.int32)

        num_te_batch = 2405 // batch_size
        return teX, teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY

def load_sdugait_img(batch_size, is_training):
    data = np.load('data/sdugait/docs.npy')
    data = np.resize(data[:,1:],(24180,2025))
    print(data.shape) 
    label = np.loadtxt('data/sdugait/labels.txt')
    label=label[:,1]
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((19344, 45,45, 1)).astype(np.float32)
        trainY = trainY.reshape((19344)).astype(np.int32)	
        trX = trainX[:17344]
        trY = trainY[:17344]

        valX = trainX[17344:,]
        valY = trainY[17344:]

        num_tr_batch = 17344 // batch_size
        num_val_batch = 2000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((4836, 45,45, 1)).astype(np.float32)
        teY = teY.reshape((4836)).astype(np.int32)

        num_te_batch = 4836 // batch_size
        return teX, teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY




def load_ndds(batch_size, is_training):
    data = np.loadtxt('data/ndds/tsfuse/lrts10sdata12fea.txt')
    label = np.loadtxt('data/ndds/tsfuse/lrts10slabel12fea.txt')
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((604, 12, 10, 1)).astype(np.float32)
        trainY = trainY.reshape((604)).astype(np.int32)	
        trX = trainX[:550] 
        trY = trainY[:550]

        valX = trainX[550:,] 
        valY = trainY[550:]

        num_tr_batch = 550 // batch_size
        num_val_batch = 54 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((152,  12, 10, 1)).astype(np.float32)
        teY = teY.reshape((152)).astype(np.int32)

        num_te_batch = 152 // batch_size
        return teX, teY, num_te_batch
    
    
#    data = np.loadtxt('data/ndds/txtfuse/lr10sdata.txt')
#    label = np.loadtxt('data/ndds/txtfuse/lr10slabel.txt')
#    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
#    print(trainX.shape,teX.shape)
#
#    if is_training:
#        trainX = trainX.reshape((9223, 60, 10, 1)).astype(np.float32)
#        trainY = trainY.reshape((9223)).astype(np.int32)	
#        trX = trainX[:8223] 
#        trY = trainY[:8223]
#
#        valX = trainX[8223:,] 
#        valY = trainY[8223:]
#
#        num_tr_batch = 8223 // batch_size
#        num_val_batch = 1000 // batch_size
#
#        return trX, trY, num_tr_batch, valX, valY, num_val_batch
#		
#    else:
#        teX = teX.reshape((2306,  60, 10, 1)).astype(np.float32)
#        teY = teY.reshape((2306)).astype(np.int32)
#
#        num_te_batch = 2306 // batch_size
#        return teX, teY, num_te_batch
#
#    
#
    if is_training:
        return trX, trY
    else:
        return teX, teY

def load_kinectunito(batch_size, is_training):
    data = np.loadtxt('data/unito/KINECTUNITO.txt')
    label = np.loadtxt('data/unito/UNITOLABEL.txt')
    data = data[:,0:63]
    label=label-1
    trainX,teX,trainY,teY = train_test_split(data,label,test_size=0.2)
    print(trainX.shape,teX.shape)

    if is_training:
        trainX = trainX.reshape((27626, 7, 9, 1)).astype(np.float32)
        trainY = trainY.reshape((27626)).astype(np.int32)	
        trX = trainX[:25000]
        trY = trainY[:25000]

        valX = trainX[25000:,]
        valY = trainY[25000:]

        num_tr_batch = 25000 // batch_size
        num_val_batch = 2626 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
		
    else:
        teX = teX.reshape((6907, 7, 9, 1)).astype(np.float32)
        teY = teY.reshape((6907)).astype(np.int32)

        num_te_batch = 6907 // batch_size
        return teX, teY, num_te_batch

    

    if is_training:
        return trX, trY
    else:
        return teX, teY
  


def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'kinectunito':
        return load_kinectunito(batch_size, is_training)
    elif dataset == 'ndds':
        return load_ndds(batch_size, is_training)
    elif dataset == 'sdugait':
        return load_sdugait(batch_size, is_training)
    elif dataset == 'ju':
        return load_ju(batch_size, is_training)
    elif dataset == 'si':
        return load_si(batch_size, is_training)
    elif dataset == 'ga':
        return load_ga(batch_size, is_training)
    elif dataset == 'casia':
        return load_casia(batch_size, is_training)
    elif dataset == 'oulp':
        return load_oulp(batch_size, is_training)
    elif dataset == 'sdu-img':
        return load_sdugait_img(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'kinectunito':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_kinectunito(batch_size, is_training=True)
    elif dataset == 'ndds':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_ndds(batch_size, is_training=True)
    elif dataset == 'sdugait':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_sdugait(batch_size, is_training=True)
    elif dataset == 'ju':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_ju(batch_size, is_training=True)
    elif dataset == 'si':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_si(batch_size, is_training=True)
    elif dataset == 'ga':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_ga(batch_size, is_training=True)
    elif dataset == 'casia':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_casia(batch_size, is_training=True)
    elif dataset == 'oulp':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_oulp(batch_size, is_training=True)
    elif dataset == 'sdu-img':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_sdugait_img(batch_size, is_training=True)
 
        
    data_queues = tf.train.slice_input_producer([trX, trY])
#    X, Y = tf.data.Dataset.shuffle(min_after_dequeue).batch(batch_size)
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
