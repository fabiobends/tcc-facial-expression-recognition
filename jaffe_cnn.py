import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import image
import os

mpl.rc('figure', max_open_warning = 0)

class Jaffe():

  def __init__(self):
    self.dataset = {}
    self.root = 'jaffe/convolutional_network'

  def data(self, verbose = False):
    if(verbose):
      print("Generating images...")
    persons = ['KA','KL','KM','KR','MK','NA','NM','TM','UY','YM']
    emotions = ['AN','DI','FE','HA','NE','SA','SU']
    #AN = ANGRY; #DI = DISGUST; #FE = FEAR; #HA = HAPINESS;
    #NE = NEUTRAL; #SA = SADNESS; #SU = SURPRISE;
    repetitions = 5
    loop = 0
    # for process 1, remove this "-10"
    directions = [10,-10]
    start_crop = 20
    end_crop = 200

    images = []
    labels = []
    dataset = {}

    for person in persons:
      for emotion in emotions:
        for i in range(repetitions):
          if os.path.exists('datasets/jaffedbase/no_numbers/'+person+'.'+emotion+str(i+1)+'.tiff'):
            loop = loop + 1
            img = image.imread('datasets/jaffedbase/no_numbers/'+person+'.'+emotion+str(i+1)+'.tiff')
            img = np.roll(img, -15, axis=1) # np.roll(img, 10, axis=1) shifted to the left 10 pixels
            img = np.roll(img, -25, axis=0) # shifted to the top 25 pixels
            
            images.append(img[start_crop:end_crop,start_crop:end_crop]) # cropping image between 20 until 199 pixels
            labels.append(emotion)

            for d in directions:
              img2 = np.roll(img, d, axis=1).copy()   
              img2 = img2[start_crop:end_crop,start_crop:end_crop] 

              images.append(img2)
              labels.append(emotion)
            
              # for process 1, comment these lines
              img2 = np.roll(img, d, axis=0).copy()
              img2 = img2[start_crop:end_crop,start_crop:end_crop]
              images.append(img2)
              labels.append(emotion)
            
            img = np.flip(img, 1) # flipping image horizontally
            img = np.roll(img, -25, axis=1).copy()

            images.append(img[start_crop:end_crop,start_crop:end_crop])
            labels.append(emotion)

            for d in directions:
              img2 = np.roll(img, d, axis=1).copy() 
              img2 = img2[start_crop:end_crop,start_crop:end_crop]

              images.append(img2)
              labels.append(emotion)

              # for process 1, comment these lines
              img2 = np.roll(img, d, axis=0).copy() 
              img2 = img2[start_crop:end_crop,start_crop:end_crop] 
              images.append(img2)
              labels.append(emotion)
        
    dataset['images'] = images  
    dataset['labels'] = labels

    self.dataset = dataset

    return dataset
  
  def save_some_samples(self):
    if os.path.exists(self.root) is not True:
      os.mkdir(self.root)
      print('\"' + self.root + '\" folder created')
    if os.path.exists(self.root + '/processed') is not True:
      os.mkdir(self.root + '/processed')
      print('\"' + self.root + '/processed\" folder created')
    for i in range(10):
      img = self.dataset['images'][i]
      fig = plt.figure(frameon=False)
      plt.imshow(img, cmap='gray')
      plt.axis('off')
      fig.savefig(self.root + '/processed/KA_AN1_version_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
class CNN():

  def __init__(self, data, process = 0, samples = 2130):
    self.data = data
    self.process = process
    self.maps = []
    self.kernel_size = []
    self.stride_size = []
    self.polling_size = []
    self.samples = samples
    self.verbose = True
    self.convs = []
    self.__input_con = []
    self.root = 'jaffe/convolutional_network'
    self.__conv_settings()

  def __set_input_con(self,__input):
    self.__input_con = __input

  def __get_input_con(self):
    return self.__input_con

  def convulate(self):
    print('Transforming all the images...')

    print('All images: ' + str(self.samples))

    new_data = np.zeros((self.samples,self.__get_input_con().shape[0]))

    for sample in range(self.samples):
      print('Sample: ' + str(sample))
      new_data[sample] = self.__convolution(self.data['images'][sample])
      if sample == 0:
        self.save_some_convs(self.data['images'][sample])

    self.__save_convolution(new_data)
    print('Trasformation done!')

  def __convolution(self,sample):
    if self.verbose == True:
      print('Convolutional process going on...')

    convs = self.convs

    for l in range(len(self.maps)):
      if l == 0:
        __input = sample - 0.5
      else:
          if self.polling_size[l-1] > 0:
            __input = convs['polling'+str(l-1)]
          else:   
            __input = convs['layer'+str(l-1)]
      [rows, cols, pages] = convs['layer'+str(l)].shape
      for r in range(rows):
        add_r = r*self.stride_size[l]
        for c in range(cols):
          add_c = c*self.stride_size[l]
          if l == 0:
            chunk = __input[add_r:self.kernel_size[l]+add_r, add_c:self.kernel_size[l]+add_c][0]
          else:   
            chunk = __input[add_r:self.kernel_size[l]+add_r, add_c:self.kernel_size[l]+add_c]
          for m in range(self.maps[l]):
            weights = convs['w'+str(l)][m] 
            convs['layer'+str(l)][r,c,m] = np.sum(chunk*weights)

      convs['layer'+str(l)] = self.__ReLU(convs['layer'+str(l)])
      
      # for m in range(self.maps[l]):
      #   convs['layer'+str(l)][:,:,m] = convs['layer'+str(l)][:,:,m]/convs['layer'+str(l)][:,:,m].max()

      if self.polling_size[l] > 0:
        [rows_p, cols_p, pages_p] = convs['polling'+str(l)].shape
        for r in range(rows_p):
          add_r = r*self.polling_size[l]
          for c in range(cols_p):
            add_c = c*self.polling_size[l]
            for p in range(pages_p):
              chunk = convs['layer'+str(l)][add_r:add_r+self.polling_size[l], add_c:add_c+self.polling_size[l],p]
              convs['polling'+str(l)][r,c,p] = np.max(chunk)

    if(self.polling_size[-1] > 0):
      word = 'polling'
    else:
      word = 'layer'   
    last = len(self.maps) - 1
    __input_con = convs[word+str(last)].flatten()
          
    self.convs = convs

    return __input_con

  def __ReLU(self,data):
    result = np.where(data<=0.0, 0.0, data)
    return result

  def __conv_settings(self):
    print('Convolutional settings...')

    maps = [6,4,2]
    kernel_size = [6,6,2] 
    stride_size = [2,2,1]
    polling_size = [0,2,2]
    sample = self.data['images'][0][:,:,0]
    convs = {} 

    for l in range(len(maps)):
      if l == 0:
        __input = sample.reshape((180,180,1))
      else:
        if polling_size[l-1] > 0:
          __input = convs['polling'+str(l-1)]
        else:   
          __input = convs['layer'+str(l-1)]
      [rows, cols, pages] = __input.shape
      mapping_size = (rows - kernel_size[l])//stride_size[l] + 1
      m_size = (rows - kernel_size[l])/stride_size[l] + 1
      if m_size != mapping_size:
        print('Bad kernel size for map ' + str(l))  
      convs['w'+str(l)] = np.random.randn(maps[l],kernel_size[l],kernel_size[l],pages)  
      convs['layer'+str(l)] = np.zeros((mapping_size,mapping_size,maps[l]))
      if (polling_size[l] > 0):
        size = round(mapping_size/polling_size[l])
        size_ = mapping_size/polling_size[l]
        if size != size_:
          print('Bad polling size for map ' + str(l)) 
        convs['polling'+str(l)] = np.zeros((size,size,maps[l]))

    #verifying the input shape for connected network
    if(polling_size[-1] > 0):
      word = 'polling'
    else:
      word = 'layer'   
    last = len(maps) - 1
    __input_con = convs[word+str(last)].flatten()

    self.convs = convs
    self.maps = maps
    self.kernel_size = kernel_size
    self.stride_size = stride_size
    self.polling_size = polling_size
    self.__set_input_con(__input_con)
    print('Output\'s length: ' + str(__input_con.shape[0]))

  def save_some_convs(self,sample):
    
    convs = self.convs
    
    if os.path.exists(self.root + '/transformed') is not True:
      os.mkdir(self.root + '/transformed')
      print('\"' + self.root + '/transformed\" folder created')   

    if os.path.exists(self.root + '/transformed/process_' + str(self.process)) is not True:
      os.mkdir(self.root + '/transformed/process_' + str(self.process))
      print('\"' + self.root + '/transformed/process_' + str(self.process) + '\" folder created')   

    if os.path.exists(self.root + '/transformed/process_' + str(self.process) + '/convs') is not True:
      os.mkdir(self.root + '/transformed/process_' + str(self.process) + '/convs')
      print('\"' + self.root + '/transformed/process_' + str(self.process) + '/convs\" folder created')

    if os.path.exists(self.root + '/transformed/process_' + str(self.process) + '/polls') is not True:
      os.mkdir(self.root + '/transformed/process_' + str(self.process) + '/polls')
      print('\"' + self.root + '/transformed/process_' + str(self.process) + '/polls\" folder created')

    fig = plt.figure(frameon=False)
    plt.imshow(sample, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    fig.savefig(self.root + '/transformed/process_' + str(self.process) + '/input.png')
    for l in range(len(self.maps)):
      print('Convolutional layer '+str(l))
      for m in range(self.maps[l]):
        fig = plt.figure(frameon=False)
        plt.imshow(convs['layer'+str(l)][:,:,m], cmap='gray')
        plt.axis('off')
        plt.colorbar()
        fig.savefig(self.root + '/transformed/process_' + str(self.process) + '/convs/conv_layer_' + str(l) + '_map' + str(m) + '.png')
      if self.polling_size[l] > 0:
        print('Polling layer '+str(l))
        for m in range(self.maps[l]):
          fig = plt.figure(frameon=False)
          plt.imshow(convs['polling'+str(l)][:,:,m], cmap='gray') 
          plt.axis('off')
          plt.colorbar()
          fig.savefig(self.root + '/transformed/process_' + str(self.process) + '/polls/poll_layer_' + str(l) + '_map' + str(m) + '.png')
  
  def __save_convolution(self,data):
    print('Saving CSV file...')
    samples = self.samples
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    if os.path.exists(self.root + '/csv') is not True:
      os.mkdir(self.root + '/csv')
      print('\"' + self.root + '/csv\" folder created')
    if os.path.exists(self.root + '/csv/process_' + str(self.process)) is not True:
      os.mkdir(self.root + '/csv/process_' + str(self.process))
      print('\"' + self.root + '/csv/process_' + str(self.process) + '\" folder created')
    for i in range(samples):
      df['sample_'+str(i)] = data[i]
      df2['sample_'+str(i)] = [self.data['labels'][i],self.data['labels'][i]]
    df.to_csv(self.root + '/csv/process_' + str(self.process) + '/jaffe_samples.csv')
    df2.to_csv(self.root + '/csv/process_' + str(self.process) + '/jaffe_labels.csv')    
  
    print('CSV file has been saved!')

jaffe = Jaffe()
dataset = jaffe.data()
jaffe.save_some_samples()

cnn = CNN(data = dataset, samples = len(dataset['labels']), process = 0)
cnn.convulate()