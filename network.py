import numpy as np
from collections import OrderedDict
from calclation import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss,BatchNormalization

class main_network:


    def __init__(self):

        #config
        #Convolution Filter
        self.f_size = 3
        self.f_pad = 1
        self.f_stride = 1
        self.filter_num_list = [32,64,64]
        #Pooling
        self.p_size = 2
        self.p_pad = 0 
        self.p_stride = 2
        #hidden layer
        size_hidden=100

        dim_input = 1
        size_input = 28
        size_output = 15
        std = 0.01

        #initializing params
        self.params = {}
        for index in range(3):
            #calc size after conv/pooling
            size_after_conv = ( size_input + 2*self.f_pad - self.f_size) // self.f_stride + 1
            size_after_pool = (size_after_conv + 2*self.p_pad - self.p_size) // self.p_stride + 1

            #params
            self.params['W'+ str(index)] = np.random.randn(self.filter_num_list[index], dim_input, self.f_size, self.f_size) * std 
            self.params['b'+ str(index)] = np.ones(self.filter_num_list[index]) 
            self.params['gamma'+ str(index)] = np.ones(self.filter_num_list[index])
            self.params['beta'+ str(index)] = np.ones(self.filter_num_list[index])

            #updating size/dim 
            size_input = size_after_pool
            dim_input = self.filter_num_list[index]
            
        #Number of pixels in hidden layer
        pixel_in_hidden = self.filter_num_list[2] * size_after_pool * size_after_pool

        self.params['W_hidden'] = std *  np.random.randn(pixel_in_hidden, size_hidden)
        self.params['b_hidden'] = np.zeros(size_hidden)
        self.params['W_last'] = std *  np.random.randn(size_hidden, size_output)
        self.params['b_last'] = np.zeros(size_output)
        self.params['gamma_last'] = np.ones(size_hidden)
        self.params['beta_last'] = np.zeros(size_hidden)

        #layers
        self.layers = OrderedDict()
        for index in range(3):
            self.layers['Conv'+str(index)] = Convolution(self.params['W'+ str(index)], self.params['b'+ str(index)],self.f_stride, self.f_pad)
            self.layers['BatchNorm'+ str(index)] = BatchNormalization(self.params['gamma'+ str(index)], self.params['beta'+ str(index)])
            self.layers['ReLU'+ str(index)] = ReLU()
            self.layers['Pool'+ str(index)] = MaxPooling(pool_h= self.p_size, pool_w= self.p_size, stride=self.p_stride, pad=self.p_pad)
        self.layers['Affine_hidden'] = Affine(self.params['W_hidden'], self.params['b_hidden'])
        self.layers['BatchNorm_last'] = BatchNormalization(self.params['gamma_last'], self.params['beta_last'])
        self.layers['ReLU_last'] = ReLU()
        self.layers['Affine_last'] = Affine(self.params['W_last'], self.params['b_last'])

        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x,train_flg=False):
        for key, layer in self.layers.items():
            if "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        return x

    def loss(self, x, t,train_flg=False):
        y = self.predict(x,train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx,train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t,train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # config
        grads = {}
        for idx in range(3):
            grads['W'+ str(idx)], grads['b'+ str(idx)] = self.layers['Conv'+ str(idx)].dW, self.layers['Conv'+ str(idx)].db
            grads['gamma'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dgamma
            grads['beta'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dbeta
        
        grads['W_hidden'], grads['b_hidden'] = self.layers['Affine_hidden'].dW, self.layers['Affine_hidden'].db
        grads['gamma_last'] = self.layers['BatchNorm_last'].dgamma
        grads['beta_last'] = self.layers['BatchNorm_last'].dbeta
        grads['W_last'], grads['b_last'] = self.layers['Affine_last'].dW, self.layers['Affine_last'].db

        return grads        