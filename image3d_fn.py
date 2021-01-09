import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

mpl.rc('figure', max_open_warning = 0)

class CNN():

  def __init__(self, process, samples = 504):
    self.process = process
    if self.process == 0:
      self.samples = 504
    elif self.process == 1:
      self.samples = 1008
    else:
      self.samples = 2520 
    self.root = 'impa-image3d/convolutional_network'

  def get_info(self):
    print('Opening CSV file...')
    df1 = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/impa_samples.csv')
    df2 = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/impa_labels.csv')
    length = len(df1)
    dataset = {}
    inputs = np.zeros((self.samples,length))
    labels = []
    for i in range(self.samples):
      inputs[i] = df1['sample_'+str(i)]
      labels.append(df2['sample_'+str(i)][0])
    dataset['inputs'] = inputs
    dataset['labels'] = np.array(labels)
    print('CSV file has been read!')

    return dataset    

class FN():
  '''
    Fully connected networks
  '''
  def __init__(self, data, train_perc, verbose, process, learning_rate, type_of_loss = 0, type_of_func = 0, recover = False,  samples = 504) :
    self.process = process
    self.data = data
    self.new_data = {}
    self.verbose = verbose
    self.samples = samples
    self.recover = recover
    self.random1 = {}
    self.random2 = {}
    self.type_of_loss = type_of_loss
    self.type_of_func = type_of_func
    self.learning_rate = learning_rate
    self.max_epoch = 20000
    self.emotions = []
    self.classes = []
    self.train_perc = train_perc
    self.alpha = []
    self.neurons = []
    self.loss = []
    self.total_loss = []
    self.aux_loss = []
    self.train_data = {}
    self.test_data = {}
    self.__conns = []
    self.__output = []
    self.__sample = []
    self.__label = []
    self.root = 'impa-image3d/fully_connected_network'

    self.__settings()
    
  def __set_output(self, output):
    self.__output = output

  def __get_output(self):
    return self.__output

  def __set_conns(self,conns):
    self.__conns = conns
    if self.verbose == True:
      print('Full-connected params have been set up!')

  def __get_conns(self):
    return self.__conns

  def __set_sample(self, sample):
    self.__sample = sample

  def __get_sample(self):
    return self.__sample  

  def __set_label(self, label):
    #AN = ANGRY; #DI = DISGUST; #FE = FEAR; #HA = HAPINESS;
    #NE = NEUTRAL; #SA = SADNESS; #SU = SURPRISE;
    if label == 'AN':
      self.__label = 0 
    elif label == 'DI':
      self.__label = 1
    elif label == 'FE':
      self.__label = 2
    elif label == 'HA':
      self.__label = 3
    elif label == 'NE':
      self.__label = 4 
    elif label == 'SA':
      self.__label = 5
    else:
      self.__label = 6         

  def __get_label(self):
    return int(self.__label)

  def __save_info(self):
    print('Saving CSV file containing the weights...')
    # self.aux_loss.append(self.total_loss)
    conns = self.__get_conns()
    if os.path.exists(self.root) is not True:
      os.mkdir(self.root)
      print('\"' + self.root + '\" folder created')
    if os.path.exists(self.root + '/csv') is not True:
      os.mkdir(self.root + '/csv')
      print('\"' + self.root + '/csv\" folder created')  
    if os.path.exists(self.root + '/csv/process_' + str(self.process)) is not True:
      os.mkdir(self.root + '/csv/process_' + str(self.process))
      print('\"' + self.root + '/csv/process_' + str(self.process) + '\" folder created')
    if os.path.exists(self.root + '/csv/process_' + str(self.process) + '/weights') is not True:
      os.mkdir(self.root + '/csv/process_' + str(self.process) + '/weights')
      print('\"' + self.root + '/csv/process_' + str(self.process) + '/weights\" folder created')  
    for i in range(len(self.neurons)):
      df = pd.DataFrame()
      [columns, rows] = conns['w'+str(i)].shape   
      for j in range(columns):
        df['w'+str(j)] = conns['w'+str(i)][j]  
      df.to_csv(self.root + '/csv/process_' + str(self.process) + '/weights/layer'+str(i)+'.csv')
    if os.path.exists(self.root + '/csv/process_' + str(self.process) + '/random0.csv') is not True:
      df_rnd1 = {}
      for csv in range(self.classes):
        temp = {'rnd'+str(csv): self.random1['rnd'+self.emotions[csv]]}
        df_rnd1['rnd'+str(csv)] = pd.DataFrame(temp)
        df_rnd1['rnd'+str(csv)].to_csv(self.root + '/csv/process_' + str(self.process) + '/random' + str(csv) + '.csv')
    if os.path.exists(self.root + '/csv/process_' + str(self.process) + '/random_after0.csv') is not True:
      df_rnd2 = {}
      temp = {'rnd0': self.random2['rnd0']}
      df_rnd2['rnd0'] = pd.DataFrame(temp)
      df_rnd2['rnd0'].to_csv(self.root + '/csv/process_' + str(self.process) + '/random_after0.csv')
      temp2 = {'rnd1': self.random2['rnd1']}
      df_rnd2['rnd1'] = pd.DataFrame(temp2)
      df_rnd2['rnd1'].to_csv(self.root + '/csv/process_' + str(self.process) + '/random_after1.csv')
    if os.path.exists(self.root + '/plots') is not True:
      os.mkdir(self.root + '/plots')
      print('\"' + self.root + '/plots\" folder created')  
    print('CSV file has been saved!')

  def __save_loss(self):
    print('Saving CSV file containing the loss...')
    df_loss = pd.DataFrame()
    df_loss['loss'] = self.aux_loss
    df_loss.to_csv(self.root + '/csv/process_' + str(self.process) + '/loss.csv')
    print('CSV file has been saved!')

  def __save_epoch(self):
    print('Saving CSV file...')
    df_epoch = pd.DataFrame()
    df_epoch['max_epoch'] = [self.max_epoch, self.max_epoch]
    df_epoch.to_csv(self.root + '/csv/process_' + str(self.process) + '/max_epoch.csv')
    print('CSV file has been saved!')

  def __get_info(self):
    print('Getting previous info (weights and random permutations)...')
    conns = self.__get_conns()
    for i in range(len(self.neurons)):
      df = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/weights/layer'+str(i)+'.csv')
      [columns, rows] = conns['w'+str(i)].shape
      for j in range(columns):
        conns['w'+str(i)][j] = df['w'+str(j)]       
    self.__set_conns(conns)
    for i in range(self.classes):
      df_rnd1 = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/random' + str(i) + '.csv')
      self.random1['rnd'+self.emotions[i]] = np.array(df_rnd1['rnd'+str(i)])
    for i in range(2):
      df_rnd2 = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/random_after' + str(i) + '.csv')
      self.random2['rnd'+str(i)] = np.array(df_rnd2['rnd'+str(i)])  
    self.split_data()
    print('CSV file has been read!')
  
  def __get_loss(self):
    print('Getting previous info (loss)...')
    df_loss = pd.read_csv(self.root + '/csv/process_' + str(self.process) + '/loss.csv')
    self.aux_loss = list(df_loss['loss'])
    self.total_loss = self.aux_loss[-1]
    print('CSV file has been read!')

  def get_epoch(self):
    __file = self.root + '/csv/process_' + str(self.process) + '/max_epoch.csv'
    print('Getting previous info...')
    if os.path.exists(__file) is True:
      df_epoch = pd.read_csv(__file)
      self.max_epoch = list(df_epoch['max_epoch'])[0]
      return True
    else:
      return False

  def split_data(self):
    print('Splitting data...')
  
    # train_size = round(self.train_perc*self.samples)
    
    self.data['inputs'] = self.data['inputs']/self.data['inputs'].max()

    # self.new_data['inputs'] = self.data['inputs'][self.rnd]
    # self.new_data['labels'] = self.data['labels'][self.rnd]
    self.new_data = {}
    self.new_data['inputs'] = self.data['inputs']
    self.new_data['labels'] = self.data['labels']

    _data = {}
    train_data = {}
    train_data['inputs'] = []
    train_data['labels'] = []

    test_data = {}
    test_data['inputs'] = []
    test_data['labels'] = []

    for emotion in self.emotions:
      _data[emotion] = []

    for num in range(len(self.new_data['labels'])):
      label = self.new_data['labels'][num]
      _data[label].append(self.new_data['inputs'][num])

    for emotion in self.emotions:
      samples = len(_data[emotion])
      rnd = self.random1['rnd'+ emotion]
      _data[emotion] = np.array(_data[emotion])[rnd]
      train_size = round(self.train_perc*samples)
      train_data['inputs'] = train_data['inputs'] + _data[emotion].tolist()[:train_size]
      train_data['labels'] = train_data['labels'] + [emotion]*train_size
      test_data['inputs'] = test_data['inputs'] + _data[emotion].tolist()[train_size:]
      test_data['labels'] = test_data['labels'] + [emotion]*(samples-train_size)
    
    rnd1 = self.random2['rnd0']

    train_data['inputs'] = np.array(train_data['inputs'])[rnd1]
    train_data['labels'] = np.array(train_data['labels'])[rnd1]

    rnd2 = self.random2['rnd1']

    test_data['inputs'] = np.array(test_data['inputs'])[rnd2]
    test_data['labels'] = np.array(test_data['labels'])[rnd2]

    self.train_data = train_data
    self.test_data = test_data

  def train(self):
    print('Training...')
    samples = self.train_data['inputs'].shape[0]
    train = True

    if self.recover == True:
      if os.path.exists(self.root + '/csv/process_'+str(self.process)) is not True:
        epoch = 0
      else:   
        self.__get_info()
        self.__get_loss()
        epoch = len(self.aux_loss)
    else: 
      epoch = 0

    min_error = 0.001

    if train == True:
      while epoch < self.max_epoch and self.total_loss > min_error:
        for sample in range(samples):
          self.__set_sample(self.train_data['inputs'][sample])
          self.__set_label(self.train_data['labels'][sample])
          self.__forward()
          self.__backward()
          self.total_loss = np.sum(self.loss)/self.loss.shape[0]
        self.aux_loss.append(self.total_loss)
        print(epoch)
        print(self.loss)
        print(self.total_loss)  
        epoch += 1 
        if epoch == 1:
          self.__save_info()
          self.__save_loss()   
        if epoch % 50 == 0:
          if self.total_loss <= np.min(self.aux_loss):
            self.__save_info()
          self.__save_loss()
          fig = plt.figure()
          plt.plot(np.arange(epoch),self.aux_loss)
          plt.xlabel('Épocas')
          plt.ylabel('Perda')
          plt.title('Curva de treinamento')
          fig.savefig(self.root + '/plots/training_curve_'+ str(self.process) + '.png')
    
    self.__save_epoch()

    if self.total_loss <= np.min(self.aux_loss):
      self.__save_info()

    fig = plt.figure()
    plt.plot(np.arange(epoch),self.aux_loss)
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Curva de treinamento')
    fig.savefig(self.root + '/plots/training_curve_'+ str(self.process) + '.png')  

  def test(self):
    print('Testing...')
    
    self.__get_info()

    samples = self.test_data['inputs'].shape[0]
    self.conf_matrix = np.zeros((self.classes,self.classes))

    for i in range(samples):
      sample = self.test_data['inputs'][i] 
      label = self.test_data['labels'][i]
      self.__set_sample(sample)
      self.__set_label(label)
      self.__forward() 
      result = self.__get_output()

      if self.type_of_func == 0:
        self.loss = (1 - self.loss)**2/2
      else:  
        prob = self.__softmax(result)
        if self.type_of_loss == 0:
          self.loss = - np.log(prob)
        else: 
          self.loss = 1 - prob
      likely = np.argmin(self.loss)
      label = self.__get_label()
      self.conf_matrix[likely][label] += 1

    print('Confusion matrix:')
    print(self.conf_matrix)
    print('Hits:')  
    num = 0
    for i in range(self.classes):
      num += self.conf_matrix[i][i]
    print(num/np.sum(self.conf_matrix))
    
  def __forward(self):
    if self.verbose == True:
      print('Forward process...')

    __input = self.__get_sample()
    conns = self.__get_conns()

    if self.type_of_func == 0:
      for i in range(len(self.neurons)):
        if i != 0:
          __input = conns['o'+str(i-1)]
        input_bias = np.append(np.array(1),__input)
        prod = input_bias.dot(conns['w'+str(i)])
        conns['o'+str(i)] = self.__logistic(prod)
    else:  
      for i in range(len(self.neurons)):
        if i != 0:
          __input = conns['o'+str(i-1)]
        input_bias = np.append(np.array(1),__input)
        prod = input_bias.dot(conns['w'+str(i)])
        conns['o'+str(i)] = self.__ReLU(prod)

    last = len(self.neurons) - 1
    self.__set_conns(conns)
    self.__set_output(conns['o'+str(last)])

  def __backward(self):
    if self.verbose == True:
      print('Backward process...')
      print('Connected process coming back...')

    label = self.__get_label()
    conns = self.__get_conns()    

    if self.type_of_func == 0:
      for i in reversed(range(len(self.neurons))):
        last = len(self.neurons)-1
        __grad = np.zeros(conns['grad'+str(i)].shape[0]) 
        if i == last:
          result = conns['o'+str(last)]
          for c in range(self.neurons[last]):
            if c == label:
              self.loss[c] = (1 - result[c])**2/2
              __grad[c] = (1 - result[c])*(1-result[c])*result[c]  # prob[label]-1 / #-prob[label]*(1-prob[label])
            else:
              __grad[c] = (0 - result[c])*(1-result[c])*result[c] # prob[label] / #prob[label]*prob[c]
          __input = conns['o'+str(i-1)]
          # print(__grad.shape)
          der = (1-__input)*__input
          prod_w = np.outer(der,__grad)

          conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - self.alpha*prod_w
          prod_b = 1*__grad
          conns['w'+str(i)][0] -= self.alpha*prod_b
        else:
          if i == 0:
            __input = self.__get_sample()
          else:  
            __input = conns['o'+str(i-1)]
          prev_grad = conns['grad'+str(i+1)]
          __grad = prev_grad.dot(conns['w'+str(i+1)][1:].T)
          # print(__grad.shape)
          der = (1-__input)*__input
          prod_w = np.outer(der,__grad)

          conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - self.alpha*prod_w
          prod_b = 1*__grad
          conns['w'+str(i)][0] -= self.alpha*prod_b
          # upstreem gradiente
        conns['grad'+str(i)] = __grad

    else:  
      for i in reversed(range(len(self.neurons))):
        last = len(self.neurons)-1
        __grad = np.zeros(conns['grad'+str(i)].shape[0]) 
        if i == last:
          result = conns['o'+str(last)]
          prob = self.__softmax(result)
          for c in range(self.neurons[last]):
            if c == label:
              if self.type_of_loss == 0:
                self.loss[c] = - np.log(prob[c])
              else:  
                self.loss[c] = 1 - prob[c]
              __grad[c] = prob[label] - 1  # prob[label]-1 / #-prob[label]*(1-prob[label])
            else:
              if self.type_of_loss == 0:
                __grad[c] = prob[label]  
              else:  
                __grad[c] = prob[c] # prob[label] / #prob[label]*prob[c]
          __input = conns['o'+str(i-1)]
          # print(__grad.shape)
          prod_w = np.outer(__input,__grad)
          conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - (self.alpha/10**i)*prod_w
          prod_b = 1*__grad
          conns['w'+str(i)][0] -= self.alpha*prod_b
        else:
          if i == 0:
            __input = self.__get_sample()
          else:  
            __input = conns['o'+str(i-1)]
          prev_grad = conns['grad'+str(i+1)]
          __grad = prev_grad.dot(conns['w'+str(i+1)][1:].T)
          # print(__grad.shape)
          if __input.max() != 0:
            prod_w = np.outer(__input/__input.max(),__grad)
          else:
            prod_w = np.outer(__input,__grad)
          conns['w'+str(i)][1:] = conns['w'+str(i)][1:] - self.alpha*prod_w
          prod_b = 1*__grad
          conns['w'+str(i)][0] -= self.alpha*prod_b
          # upstreem gradiente
        conns['grad'+str(i)] = __grad

    self.__set_conns(conns)    

  def __ReLU(self,data):
    result = np.where(data<=0.0, 0.0, data)
    return result

  def __logistic(self,data):
    result = 1/(1+np.exp(-data))
    return result

  def __softmax(self,vector):
    exps = np.exp(vector.tolist())
    return exps/np.sum(exps)     
   
  def __settings(self):
    print('Full-connected settings...')

    self.emotions = sorted(list(set(self.data['labels'])))
    self.classes = len(self.emotions)

    classes = self.classes
    if self.learning_rate == 0:
      alpha = 0.001
    elif self.learning_rate == 1:
      alpha = 0.0005
    elif self.learning_rate == 2:
      alpha = 0.00025
    else:
      alpha = 0.0001

    if self.process == 0:
      self.samples = 504
    elif self.process == 1:
      self.samples = 1008
    else:
      self.samples = 2520  

    neurons = [10,classes]
    perc_connection = 1.0
    sample = self.data['inputs'][0] #the first sample
    conns = {}
    __input_con = sample

    for i in range(len(neurons)):
      if (i == 0):
        conns['w'+str(i)] = np.random.randn(__input_con.shape[0]+1, neurons[i])
        connected = np.floor(neurons[i]*perc_connection)
        non_connected = int(neurons[i]-connected)
        for j in range(__input_con.shape[0]+1):
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

    self.__set_conns(conns)  
    self.alpha = alpha
    self.neurons = neurons
    self.loss = np.zeros(classes)
    self.loss.fill(1)
    self.total_loss = np.sum(self.loss)/self.loss.shape[0]
    self.rnd = np.random.permutation(self.samples)

    _data = {}
    train_data = {}
    train_data['labels'] = []
    test_data = {}
    test_data['labels'] = []

    for emotion in self.emotions:
      _data[emotion] = []

    for num in range(len(self.data['labels'])):
      label = self.data['labels'][num]
      _data[label].append(self.data['inputs'][num])

    for emotion in self.emotions:
      samples = len(_data[emotion])
      rnd = np.random.permutation(samples)
      self.random1['rnd'+ emotion] = rnd
      _data[emotion] = np.array(_data[emotion])[rnd]
      train_size = round(self.train_perc*samples)
      train_data['labels'] = train_data['labels'] + [emotion]*train_size
      test_data['labels'] = test_data['labels'] + [emotion]*(samples-train_size)
    
    len1 = len(train_data['labels'])
    self.random2['rnd0'] = np.random.permutation(len1)

    len2 = len(test_data['labels'])
    self.random2['rnd1'] = np.random.permutation(len2)

    print('Everything is ready.') 
  
  def save_a_sample(self):
    fig = plt.figure(frameon=False)
    plt.imshow(self.data['inputs'][0].reshape([-1,1]), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    fig.savefig('impa-image3d/convolutional_network/transformed/process_' + str(self.process) + '/output.png')

process = int(input('Enter the process number: '))
type_of_func = int(input('Type of activation function: (0 - logistic) | (1 - ReLU): '))
type_of_loss = 0
if type_of_func != 0:
  type_of_loss = int(input('Type of loss: (0 - ln) | (1 - simple): '))
learning_rate = int(input('Learning rate: (0 - 0.001) | (1 - 0.0005) | (2 - 0.00025) | (3 - 0.0001): '))
test = CNN(process = process)
data = test.get_info()     
net = FN(data = data, train_perc = 0.7, verbose = False, learning_rate = learning_rate, type_of_loss = type_of_loss, type_of_func = type_of_func, recover = True, process = process)
prev = net.get_epoch()
if prev:
  add = int(input('There is a finished process up to the epoch ' + str(net.max_epoch) + '. How many thousands of epochs do you want to add: '))
  net.max_epoch = net.max_epoch + add*1000
net.split_data()
net.train()
net.test()