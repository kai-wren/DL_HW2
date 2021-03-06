from __future__ import print_function
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from utils import get_loss, get_random_batch, images2batches, init_uniform, relu, identity_func, mass_imshow


BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225 # IMG_SIZE*IMG_SIZE
P = 75 # D /// 3
LEARNING_RATE = 0.001
compare = True


class EncDecNetLite():
    def __init__(self):
        super(EncDecNetLite, self).__init__()
        self.w_in = np.zeros((P, D))
        self.b_in = np.zeros((1, P))
        self.w_link = np.zeros((P, P))
        self.w_rec = np.zeros((P, P))
        self.b_rec = np.zeros((1, P))
        self.w_out = np.zeros((D, P))
        self.b_out = np.zeros((1, D))
    

    def init(self):
        # initialize weights by Xavier uniform initialization
        self.w_in = init_uniform(self.w_in)
        self.w_link = init_uniform(self.w_link)
        self.w_out = init_uniform(self.w_out)
        # w_rec is special kind of weight matrix hence could be initialized as identity
        self.w_rec = np.identity(self.w_rec.shape[0])

    def forward(self, x):
        #Layer_in forward pass
        self.x = x
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P)) # [20, 75]
        self.a_in = np.matmul(self.x, self.w_in.transpose()) + B_in # [20, 75]
        z_in_numpy = relu(self.a_in)
        
        # Layer_rec forward pass
        B_rec = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_rec.reshape(1, P)) # [20, 75]
        self.a_rec = np.matmul(z_in_numpy, self.w_rec.transpose()) + B_rec # [20, 75]
        self.z_rec = relu(self.a_rec)
        
        # Layer_link forward pass
        self.x_reduce = x[:, 0::3]
        a_link = np.matmul(self.x_reduce, self.w_link.transpose())
        self.z_link = identity_func(a_link)
        
        # Layer_out forward pass
        self.z_rec_link = self.z_rec + self.z_link
        B_out = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_out.reshape(1, D)) # [20, 225]
        self.a_out = np.matmul(self.z_rec_link, self.w_out.transpose()) + B_out # [20, 225]
        y = relu(self.a_out)
        return y
    
    def vector_input_layer(self, x):
        # vector form method for performance competion
        B_in = np.matmul(np.ones((BATCH_SIZE, 1)),
                         self.b_in.reshape(1, P)) # [20, 75]
        a_in = np.matmul(x, self.w_in.transpose()) + B_in # [20, 75]
        z_in_numpy = relu(a_in)
        return z_in_numpy
        
    def scalar_input_layer(self, x):
        # scalar form method for performance competion
        z_in = np.zeros((x.shape[0], P))
        for i in range(x.shape[0]):
            a_in = np.dot(self.w_in, x[i,:].T) + self.b_in.T.flatten()
            z_in[i, :] = relu(a_in)
        return z_in

    def backprop(self, Y_batch, X_batch_train):
        dw = {}
        # Layer_out backward pass
        grad_out = 2*(Y_batch - X_batch_train)
        grad_out[self.a_out <= 0] = 0 # ReLU derivative of Layer_out
        dloss_dw_out = np.dot(grad_out.T, self.z_rec_link)
        dloss_db_out = np.sum(grad_out, axis=0, keepdims=True)
        
        # Layer_link backward pass
        grad_link = np.dot(grad_out, self.w_out)
        # here no need to differentiate through z_link since there are identity function derivative of which is 1
        dloss_dw_link = np.dot(grad_link.T, self.x_reduce)
        
        # Layer_rec backward pass
        # Layer_rec is not trainable, but we still need to find gradient of 
        # Layer_rec in order to propagate derivative to Layer_in
        grad_rec = np.dot(grad_out, self.w_out)
        grad_rec[self.a_rec <= 0] = 0 # ReLU derivative of Layer_rec
        
        # Layer_in backward pass
        grad_in = np.dot(grad_rec, self.w_rec)
        grad_in[self.a_in <= 0] = 0 # ReLU derivative of Layer_in
        dloss_dw_in = np.dot(grad_in.T, self.x)
        dloss_db_in = np.sum(grad_in, axis=0, keepdims=True)
        
        # filling output dictionary with derivatives values
        dw['dloss_dw_out'] = dloss_dw_out 
        dw['dloss_db_out'] = dloss_db_out
        dw['dloss_dw_link'] = dloss_dw_link
        dw['dloss_dw_in'] = dloss_dw_in
        dw['dloss_db_in'] = dloss_db_in
        
        return dw

    def apply_dw(self, dw):
        
        # apply gradients correction to trainable weights
        self.w_in -= LEARNING_RATE * dw['dloss_dw_in']
        self.b_in -= LEARNING_RATE * dw['dloss_db_in']
        self.w_link -= LEARNING_RATE * dw['dloss_dw_link']
        self.w_out -= LEARNING_RATE * dw['dloss_dw_out']
        self.b_out -= LEARNING_RATE * dw['dloss_db_out']


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

# execute performance competion between vector and scalar form forward pass method through Layer_in
if (compare):
    print("Compare execution time in scalar and vector forms for Layer_in:")
    start = time.perf_counter()
    for i in range(UPDATES_NUM):
        X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
        test1 = neural_network.vector_input_layer(X_batch_train)
        
    stop = time.perf_counter()
    vector_time = stop-start
    print("Execution time in vector form %.7f second(s)" %vector_time )

    
    start = time.perf_counter()
    for i in range(UPDATES_NUM):
        X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
        test2 = neural_network.scalar_input_layer(X_batch_train)
        
    stop = time.perf_counter()
    scalar_time = stop-start
    print("Execution time in scalar form %.7f second(s)" %scalar_time )
    print("Vector form method is ~%.0f times faster compared to scalar form method" 
          %(scalar_time/vector_time), "with batch size = %d\n" %BATCH_SIZE )
    
losses=[]
# Main cycle
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
        
    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)
    losses.append(loss)
    
    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(Y_batch, X_batch_train)
    
    # Correct neural network''s weights
    neural_network.apply_dw(dw)

# Plotting learning curve with title and labels
print("Plotting Learning Curve as Loss\Cost function value with regards to Number of Iterations...")
plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.title("Learning curve")
plt.xlabel("Number of iterations")
plt.ylabel("Loss\Cost function value")
plt.plot(np.arange(0, UPDATES_NUM), losses)
plt.show() 
print("\n")

# Load test images
print("Plotting test images side by side true ones vs. the ones predicted by AutoEncNetLite...")
images_test = pickle.load(open('images_test.pickle', 'rb'))
BATCH_SIZE = images_test.shape[0] # setting global variable BATCH_SIZE to the number of pictures in test set
# convert test images to the network input format
batch_test = images2batches(images_test)
# execute network forward pass on test images set
Y_test = neural_network.forward(batch_test)
# reshape network output to the grayscale images to plot it
Y_test = Y_test.reshape((images_test.shape[0], images_test.shape[1], images_test.shape[2]))
# plot original images side by side with predicted
mass_imshow(images_test, Y_test)


