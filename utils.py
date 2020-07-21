# importing required libraries
import numpy as np
import matplotlib.pyplot as plt


def diff_numpy(a, b, msg=None):
    """Shows differences between two tensors"""
    if a.shape != b.shape:
        print('Wrong shape!')
        print(a.shape)
        print(b.shape)
    else:
        diff = (np.sum(a - b))**2
        if msg:
            print('%s diff = %1.6f' % (msg, diff.item()))
        else:
            print('diff = %1.6f' % diff.item())


def images2batches(images):
    """Converts images to convenient for batching form"""
    ndata, img_size, _ = images.shape
    return np.reshape(images, (ndata, img_size*img_size))


def mass_imshow(img_true, img_predict):
    """Show image using matplotlib"""
    # re-implement this function do display both batches of images - true vs. predicted
    plt.figure(figsize=(2,20))
    for i in range(img_true.shape[0]):
        
        # plotting original image in batch
        plt.subplot(img_true.shape[0], 2, i*2+1)
        plt.imshow(img_true[i], cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        if i == 0:
            plt.title("True:")
            
        # plotting predicted image in batch
        plt.subplot(img_true.shape[0], 2, i*2+2)
        plt.imshow(img_predict[i], cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        if i == 0:
            plt.title("Predicted:")
        
    plt.show()


def init_uniform(a):
    # formula taken from here -  https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init
    lim = np.sqrt(6/(a.shape[0] + a.shape[1]))
    res = [[ np.random.uniform(-lim, lim) for i in range(a.shape[1])] for j in range(a.shape[0])]
    res = np.array(res)
    return res


def relu(m):
    # relu function implementation
    return np.array([np.maximum(x, 0) for x in m])

def identity_func(a):
    # identity function implementation for better readability, otherwise function is useless
    return a


def get_random_batch(batches_train, batch_size):
    # return random batch of train images of size batch_size
    return batches_train[np.random.randint(batches_train.shape[0], size=batch_size), :] / 255


def get_loss(Y_batch, X_batch_train):
    # calculate overall loss of the AutoEncNetLite
    return np.sum((Y_batch - X_batch_train)**2)
    
