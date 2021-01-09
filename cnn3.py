import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image

class Data_Images():
  
  def __init__(self):
    pass

  def data(self):
    data = []
    data_images = np.zeros([165,122,122,3],dtype='float')
    data_classes = np.zeros([165],dtype='uint8')
    types = ['centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy','surprised','wink'] 

    for i in range(15):
      for j in range(11):
          pos = j+11*i
          data_classes[pos] = i
          if i < 9:
              string = 'images/subject0' + str(i+1) + types[j] + '.jpg'
              if i == 5:
                  img = image.imread(string)
                  data_images[pos] = img[:,:122]/255
              else:   
                  img = image.imread(string)
                  data_images[pos] = img[:,20:142]/255
          else:
              string = 'images/subject' + str(i+1) + types[j] + '.jpg'
              if i == 11:
                  img = image.imread(string)
                  data_images[pos] = img[:,30:152]/255
              else:
                  img = image.imread(string)
                  data_images[pos] = img[:,20:142]/255

    data.append(data_images)
    data.append(data_classes)

    return data

class CNN():
  '''
    Convolucional neural networks
  '''
  def __init__(self):
    self.data = []
    self.maps = []
    self.kernel_size = []
    self.stride_size = []
    self.type_of_kernel = 'full'
    self.kernels = 0
    self.polling = []
    self.polling_size = []
    self.alpha_conv = []
    self.alpha_conn = []
    self.neurons = []
    self.loss = []
    self.total_loss = []
    self.__convs = []
    self.__conns = []
    self.__input_con = []
    self.__output_con = []
    self.__sample = []
    self.__label = []
    
  def __set_data(self,data):
    self.data = data
    print('Data has been set up!')

  def __set_convs(self,convs,verbose=False):
    self.__convs = convs
    if verbose == True:
      print('Convolutional params have been set up!')

  def __get_convs(self):
    return self.__convs

  def __set_input_con(self,input_con):
    self.__input_con = input_con

  def __get_input_con(self):
    return self.__input_con  

  def __set_conns(self,conns, verbose=False):
    self.__conns = conns
    if verbose == True:
      print('Full-connected params have been set up!')

  def __get_conns(self):
    return self.__conns

  def __get_sample(self):
    return self.__sample  

  def __set_sample(self, sample):
    self.__sample = sample

  def __get_label(self):
    return int(self.__label)

  def __set_label(self, label):
    self.__label = label

  def __set_output_con(self, output):
    self.__output_con = output

  def __get_output_con(self):
    return self.__output_con

  def train(self, data):
    print('Training...')
    self.__set_data(data)
    classes = 2
    self.__split_data(classes=classes,train_perc=0.7)
    self.__settings(classes=classes)
    samples = self.train_data[0].shape[0]
    train = True
    epoch = 0
    max_epoch = 100
    min_error = 0.001
    
    if train == True:
      while epoch < max_epoch and self.total_loss > min_error:
        for sample in range(samples):
          self.__set_sample(self.train_data[0][sample])
          self.__set_label(self.train_data[1][sample])

          self.__forward()
          self.__backward()
          self.total_loss = np.sum(self.loss)/self.loss.shape[0]
          print(self.loss)
          print(self.total_loss)
        epoch += 1  

  def __forward(self,verbose=False):
    print('Forward process...')
    self.__convolution_forward(verbose)
    self.__connected_forward(verbose)

  def __convolution_forward(self,verbose=False):
    if verbose == True:
      print('Convolutional process going on...')

    sample = self.__get_sample()
    convs = self.__get_convs()
    step = 0 # used for not-unique kernels

    for l in range(len(self.maps)):
      if l == 0:
        __input = sample
      else:
          if self.polling[l-1] == True:
            __input = convs['polling'+str(l-1)]
          else:   
            __input = convs['layer'+str(l-1)]
      [rows, cols, pages] = convs['layer'+str(l)].shape
      for r in range(rows):
        add_r = r*self.stride_size[l]
        for c in range(cols):
          add_c = c*self.stride_size[l]
          chunk = __input[add_r:self.kernel_size[l]+add_r, add_c:self.kernel_size[l]+add_c]
          for m in range(self.maps[l]):
            if (self.type_of_kernel == 'full'):
              weights = convs['w'+str(l)][m,r,c]  
            else:
              k = step%self.kernels
              weights = convs['w'+str(l)][m][k]  
            # if (r == 0) and (c == 0) and (m == 0):
            #   print('chunk size ' + str(l) + ': ' + str(chunk.shape))
            #   print('weights size ' + str(l) + ': ' + str(weights.shape))
            #   print('layer size ' + str(l) + ': ' + str(convs['layer'+str(l)].shape))
            convs['layer'+str(l)][r,c,m] = np.sum(chunk*weights)
          step += 1
        step += 1
      step = 0  

      convs['layer'+str(l)] = self.__ReLU(convs['layer'+str(l)])

      if self.polling[l] == True:
        [rows_p, cols_p, pages_p] = convs['polling'+str(l)].shape
        for r in range(rows_p):
          add_r = r*self.polling_size[l]
          for c in range(cols_p):
            add_c = c*self.polling_size[l]
            for p in range(pages_p):
              # if (r == 0) and (c == 0) and (p == 0):
              #   print('polling size ' + str(l) + ': ' + str(convs['polling'+str(l)].shape))
              chunk = convs['layer'+str(l)][add_r:add_r+self.polling_size[l], add_c:add_c+self.polling_size[l],p]
              convs['polling'+str(l)][r,c,p] = np.max(chunk)

    if(self.polling[-1] == True):
      word = 'polling'
    else:
      word = 'layer'   
    last = len(self.maps) - 1
    __input_con = convs[word+str(last)].flatten()
    __input_con = __input_con/__input_con.max()
    
    # print(convs['polling0'].shape)
    # print(convs['polling2'].shape)
    self.__set_convs(convs,verbose=verbose)        
    self.__set_input_con(__input_con)

  def __connected_forward(self,verbose=False):
    if verbose == True:
      print('Connected process going on...')
    
    __input = self.__get_input_con()
    conns = self.__get_conns()

    for i in range(len(self.neurons)):
      if i != 0:
        __input = conns['o'+str(i-1)]
      input_bias = np.append(np.array(1),__input)
      prod = input_bias.dot(conns['w'+str(i)])
      conns['o'+str(i)] = self.__ReLU(prod)

    last = len(self.neurons) - 1
    self.__set_conns(conns,verbose=verbose)
    self.__set_output_con(conns['o'+str(last)])

  def __backward(self,verbose=False):
    print('Backward process...')
    self.__connected_backward(verbose)
    self.__convolution_backward(verbose)

  def __connected_backward(self,verbose=False):
    if verbose == True:
      print('Connected process coming back...')

    label = self.__get_label()
    conns = self.__get_conns()    

    for i in reversed(range(len(self.neurons))):
      last = len(self.neurons)-1
      __grad = np.zeros(conns['grad'+str(i)].shape[0]) 
      if i == last:
        result = conns['o'+str(last)]
        prob = self.__softmax(result)
        for c in range(self.neurons[last]):
          if c == label:
            self.loss[c] = np.log(prob[c]**-1)
            __grad[c] = prob[label] - 1  #prob[label]*(1-prob[label])*(1-prob[label]) / #prob[label]-1 / #-prob[label]*(1-prob[label])
          else:
            __grad[c] = prob[c] #-prob[c]*(prob[label])*(1-prob[c]) / #prob[label] / #prob[label]*prob[c]
        __input = conns['o'+str(i-1)]
        # print(__grad.shape)
        prod_w = np.outer(__input,__grad)
        conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - self.alpha_conn*prod_w
        prod_b = 1*__grad
        conns['w'+str(i)][0] -= self.alpha_conn*prod_b
      else:
        if i == 0:
          __input = self.__get_input_con()
        else:  
          __input = conns['o'+str(i-1)]
        prev_grad = conns['grad'+str(i+1)]
        __grad = prev_grad.dot(conns['w'+str(i+1)][1:].T)
        # print(__grad.shape)
        prod_w = np.outer(__input,__grad)
        conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - self.alpha_conn*prod_w
        prod_b = 1*__grad
        conns['w'+str(i)][0] -= self.alpha_conn*prod_b
        # upstreem gradiente
      conns['grad'+str(i)] = __grad

    self.__set_conns(conns,verbose=verbose)  

  def __convolution_backward(self,verbose=False):
    if verbose == True:
      print('Convolutional process coming back...')
    
    convs = self.__get_convs()
    conns = self.__get_conns()

    last = len(self.maps)-1

    for i in reversed(range(len(self.maps))):
      if i == last:
        convs = self.__last_grad(conns=conns,convs=convs,last=last)   
        convs = self.__update_kernel(convs=convs,i=last,verbose=verbose)        
      else:
        convs = self.__other_grad(convs=convs,i=i)
        convs = self.__update_kernel(convs=convs,i=i,verbose=verbose)
        
    self.__set_convs(convs,verbose=verbose)  

  def __ReLU(self,data):
    result = np.where(data<=0.0, 0.0, data)
    return result

  def __softmax(self,vector):
    exps = np.exp(vector.tolist())
    return exps/np.sum(exps)     

  def __last_grad(self,convs,conns,last):
    # print('Last grad...')
    step = 0
    if self.polling[last] == True:
      shape = convs['pollgrad'+str(last)].shape
      __pollgrad = conns['grad'+str(0)].dot(conns['w'+str(0)][1:].T).reshape(shape)
      for j in range(shape[0]):
        addx = j*self.polling_size[last]
        for l in range(shape[1]):
          addy = l*self.polling_size[last]
          for m in range(shape[2]):
            grad = __pollgrad[j,l,m]
            convs['grad'+str(last)][addx:addx+self.polling_size[last],addy:addy+self.polling_size[last],m].fill(grad)
    else:
      shape = convs['layer'+str(last)].shape
      convs['grad'+str(last)] = conns['grad'+str(0)].dot(conns['w'+str(0)][1:].T).reshape(shape)        
    
    return convs

  def __other_grad(self,convs,i):
    # print('Convgrad '+str(i))
    if self.polling[i] == True:
      shape = convs['pollgrad'+str(i)].shape
      __pollgrad = convs['pollgrad'+str(i)]
      for j in range(shape[0]):
        addx = j*self.polling_size[i]
        for l in range(shape[1]):
          addy = l*self.polling_size[i]
          for m in range(shape[2]):
            grad = __pollgrad[j,l,m]
            convs['grad'+str(i)][addx:addx+self.polling_size[i],addy:addy+self.polling_size[i],m].fill(grad)   

    return convs
    
  def __update_kernel(self,convs,i,verbose=False):
    if verbose == True:
      print('Updating kernel...')
    step = 0
    if i == 0:
      __input = self.__get_sample()  
    elif self.polling[i-1] == True:
      word = 'pollgrad'
      __input = convs['polling'+str(i-1)]
    else:   
      word = 'grad'
      __input = convs['layer'+str(i-1)]
    [rows, cols, pages] = convs['layer'+str(i)].shape
    _max = __input.max()
    for r in range(rows):
      add_r = r*self.stride_size[i]
      for c in range(cols):
        add_c = c*self.stride_size[i]
        chunk = __input[add_r:self.kernel_size[i]+add_r, add_c:self.kernel_size[i]+add_c]
        for m in range(self.maps[i]):
          if self.type_of_kernel == 'full':
            convs['w'+str(i)][m,r,c] -= self.alpha_conv*(chunk/_max)*convs['grad'+str(i)][r,c,m]
            weights = convs['w'+str(i)][m,r,c]
            if i != 0:
              convs[word+str(i-1)][add_r:self.kernel_size[i]+add_r, add_c:self.kernel_size[i]+add_c] += weights*convs['grad'+str(i)][r,c,m]  
          else:
            k = step%self.kernels
            convs['w'+str(i)][m,k] -= self.alpha_conv*(chunk/_max)*convs['grad'+str(i)][r,c,m]
            weights = convs['w'+str(i)][m,k]
            if i != 0:
              convs[word+str(i-1)][add_r:self.kernel_size[i]+add_r, add_c:self.kernel_size[i]+add_c] += weights*convs['grad'+str(i)][r,c,m]
        step += 1
      step += 1
    step = 0    

    return convs

  def __settings(self,classes):
    # convolutional settings
    maps = [3,4,2]
    kernel_size = [4,3,2] #4x4,3x3,5x5
    stride_size = [2,1,2]
    type_of_kernel = 'full' #or 'full' and 'not-unique'
    kernels = 5 # for full=connected, that is irrelevant
    polling = [True,False,True] # polling yes or no
    polling_size = [2,0,2]
    alpha_conn = 0.001
    alpha_conv = 0.000001
    neurons = [6,4,classes]
    perc_connection = 1.0
    __input_con = []
    data = self.data
    sample = data[0][0] #the first sample
  
    convs = {} 
    
    #CONVOLUTION
    for l in range(len(maps)):
      if l == 0:
        __input = sample
      else:
        if polling[l-1] == True:
          __input = convs['polling'+str(l-1)]
        else:   
          __input = convs['layer'+str(l-1)]
      [rows, cols, pages] = __input.shape
      mapping_size = (rows - kernel_size[l])//stride_size[l] + 1
      m_size = (rows - kernel_size[l])/stride_size[l] + 1
      if m_size != mapping_size:
        print('Bad kernel size for map ' + str(l))  
      if (type_of_kernel == 'full'):    
        convs['w'+str(l)] = np.random.randn(maps[l],mapping_size,mapping_size,kernel_size[l],kernel_size[l],pages)
      else:
        convs['w'+str(l)] = np.random.randn(maps[l],kernels,kernel_size[l],kernel_size[l],pages)  
      convs['layer'+str(l)] = np.zeros((mapping_size,mapping_size,maps[l]))
      convs['grad'+str(l)] = np.zeros((mapping_size,mapping_size,maps[l]))
      if (polling[l] == True):
        size = round(mapping_size/polling_size[l])
        size_ = mapping_size/polling_size[l]
        if size != size_:
          print('Bad polling size for map ' + str(l)) 
        convs['polling'+str(l)] = np.zeros((size,size,maps[l]))
        convs['pollgrad'+str(l)] = np.zeros((size,size,maps[l]))

    #verifying the input shape for connected network
    if(polling[-1] == True):
      word = 'polling'
    else:
      word = 'layer'   
    last = len(maps) - 1
    __input_con = convs[word+str(last)].flatten()
    
    conns = {}
    
    #CONNECTED
    for i in range(len(neurons)):
      if (i == 0):
        conns['w'+str(i)] = np.random.randn(__input_con.shape[0]+1, neurons[i])
        connected = np.floor(neurons[i]*perc_connection)
        non_connected = int(neurons[i]-connected)
        for j in range(__input.shape[0]+1):
          for k in range(non_connected):
            conns['w'+str(i)][j][k] = 0.0
          ind = np.random.permutation(neurons[i])
          conns['w'+str(i)][j] = conns['w'+str(i)][j][ind]
      else:
        conns['w'+str(i)] = np.random.randn(neurons[i-1]+1, neurons[i])
        connected = np.floor(neurons[i]*perc_connection)
        non_connected = int(neurons[i]-connected)
        for j in range(neurons[i-1]+1):
          for k in range(non_connected):
            conns['w'+str(i)][j][k] = 0.0
          ind = np.random.permutation(neurons[i])
          conns['w'+str(i)][j] = conns['w'+str(i)][j][ind]

    for i in range(len(neurons)):
      conns['o'+str(i)] = np.zeros(neurons[i])
      conns['grad'+str(i)] = np.zeros(neurons[i])

    self.__set_convs(convs)
    self.__set_conns(conns)
    self.maps = maps
    self.kernel_size = kernel_size
    self.stride_size = stride_size
    self.type_of_kernel = type_of_kernel
    self.kernels = kernels
    self.polling = polling
    self.polling_size = polling_size
    self.alpha_conn = alpha_conn
    self.alpha_conv = alpha_conv
    self.neurons = neurons
    self.loss = np.zeros(classes)
    self.loss.fill(1)
    self.total_loss = np.sum(self.loss)/self.loss.shape[0]

    print('Everything is ready.')  

  def show_convs(self):
    print('Showing resulting convolutional layers...')
    
    convs = self.__get_convs()
    [sample, _] = self.__get_sample()
    print('----------------------')
    print('----------------------')
    plt.imshow(sample)
    plt.show()
    for l in range(len(self.maps)):
      print('----------------------')
      print('----------------------')
      print('Convolutional layer '+str(l))
      for m in range(self.maps[l]):
        print('Map '+str(m))
        plt.imshow(convs['layer'+str(l)][:,:,m])
        plt.colorbar()
        plt.show()
      if self.polling[l] == True:
        print('----------------------')
        print('Polling layer '+str(l))
        for m in range(self.maps[l]):
          print('Map '+str(m))
          plt.imshow(convs['polling'+str(l)][:,:,m]) 
          plt.colorbar()
          plt.show()

  def __split_data(self, classes, train_perc):
    print('Splitting data...')
    samples = []
    train_size = round(train_perc*11)
    test_size = 11 - train_size
    train_label = np.zeros(train_size*classes)
    test_label = np.zeros(test_size*classes)

    for i in range(classes):
      samples.append(self.data[0][i*11:(i+1)*11])
      train_label[i*train_size:(i+1)*train_size].fill(i)
      test_label[i*test_size:(i+1)*test_size].fill(i)
      
    for i in range(classes):
      rnd = np.random.permutation(11)
      current = samples[i][rnd]
      if i == 0:
        train_data = current[:train_size]
        test_data = current[train_size:]
      else: 
        train_data = np.append(train_data,current[:train_size],axis=0)
        test_data = np.append(test_data,current[train_size:],axis=0)   

    rnd1 = np.random.permutation(train_size*classes)
    rnd2 = np.random.permutation(test_size*classes)

    train_data = train_data[rnd1]
    train_label = train_label[rnd1]
    test_data = test_data[rnd2]
    test_label = test_label[rnd2]

    tr_data = []
    ts_data = []

    tr_data.append(train_data)
    tr_data.append(train_label)
    ts_data.append(test_data)
    ts_data.append(test_label)
    
    self.train_data = tr_data 
    self.test_data = ts_data

  def test(self):
    print('Testing...')
    classes = 2 
    
    samples = self.test_data[0].shape[0]
    self.conf_matrix = np.zeros((classes,classes))

    for i in range(samples):
      sample = self.test_data[0][i] 
      label = self.test_data[1][i] 

      self.__set_sample(sample)
      self.__forward()
      convs = self.__get_conns() 
      result = self.__get_output_con()
      prob = self.__softmax(result)
      self.loss = -np.log(prob)
      likely = np.argmin(self.loss)
      self.conf_matrix[likely][int(label)] += 1

    print('Confusion matrix:')
    print(self.conf_matrix)
    print('Hits:')  
    num = 0
    for i in range(classes):
      num += self.conf_matrix[i][i]
    print(num/np.sum(self.conf_matrix))
     
img = Data_Images()
data = img.data()     
net = CNN()
net.train(data)
net.test()