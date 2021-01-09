import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import image
import os

mpl.rc('figure', max_open_warning = 0)

class IMPA():

  def __init__(self):
    self.dataset = {}
    self.emotions = []
    self.root = 'impa-image3d/convolutional_network'

  def rgb2gray(self,rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

  def data(self, verbose = False):
    if(verbose):
      print("Generating images...")
    self.emotions = ['NE','HA','SA','SU','AN','DI','FE']
    #AN = ANGRY; #DI = DISGUST; #FE = FEAR; #HA = HAPINESS;
    #NE = NEUTRAL; #SA = SADNESS; #SU = SURPRISE;

    images = []
    labels = []
    # for process 1, remore this '-10'
    directions = [10, -10]
    dataset = {}

    for person in range(38):
      if person == 19 or person == 21:
        continue
      for emotion in range(7):

        if person < 9:
          noun = 'datasets/impa-image3d/s00'+ str(person+1) + '/tif/s00'+str(person+1)+'-0'+str(emotion)+'_img.tif'  
        else:
          noun = 'datasets/impa-image3d/s0'+ str(person+1) + '/tif/s0'+str(person+1)+'-0'+str(emotion)+'_img.tif'

        img = self.rgb2gray(image.imread(noun))
        img = np.roll(img, -25, axis=0)
        
        images.append(img[90:390,170:470]) # 300 x 300 centralized and cropped version
        labels.append(self.emotions[emotion])
      
        for d in directions:
          img2 = np.roll(img, d, axis = 1)
          images.append(img2[90:390,170:470]) # 300 x 300 centralized and cropped version
          labels.append(self.emotions[emotion])
          
          # for process 1, comment these lines
          img2 = np.roll(img, d, axis = 0)
          images.append(img2[90:390,170:470]) # 300 x 300 centralized and cropped version
          labels.append(self.emotions[emotion])
          
        img = np.flip(img, 1) # flipping image horizontally

        images.append(img[90:390,170:470]) # 300 x 300 centralized and cropped version
        labels.append(self.emotions[emotion])

        for d in directions:
          img2 = np.roll(img, d, axis = 1)
          images.append(img2[90:390,170:470]) # 300 x 300 centralized and cropped version
          labels.append(self.emotions[emotion])

          # for process 1, comment these lines
          img2 = np.roll(img, d, axis = 0)
          images.append(img2[90:390,170:470]) # 300 x 300 centralized and cropped version
          labels.append(self.emotions[emotion])
          
        
    dataset['images'] = images  
    dataset['labels'] = labels
    
    self.dataset = dataset

    return dataset
  
  def save_some_samples(self):
    if os.path.exists('impa-image3d') is not True:
      os.mkdir('impa-image3d')
      print('\"impa-image3d\" folder created')
    if os.path.exists(self.root) is not True:
      os.mkdir(self.root)
      print('\"'+ self.root + '\" folder created')  
    if os.path.exists(self.root + '/processed') is not True:
      os.mkdir(self.root + '/processed')
      print('\"'+ self.root + '/processed\" folder created')
    for i in range(10):
      img = self.dataset['images'][i]
      fig = plt.figure(frameon=False)
      plt.imshow(img, cmap='gray')
      plt.axis('off')
      fig.savefig(self.root + '/processed/s001_' + str(i) + self.dataset['labels'][i] + '.png', bbox_inches='tight', pad_inches=0)
      img = self.dataset['images'][14*i]
      fig = plt.figure(frameon=False)
      plt.imshow(img, cmap='gray')
      plt.axis('off')
      fig.savefig(self.root + '/processed/s0_' + str(i) + self.dataset['labels'][14*i] + '.png', bbox_inches='tight', pad_inches=0)  

class CNN():

  def __init__(self, data, process = 0, samples = 504):
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
    self.root = 'impa-image3d/convolutional_network'
    self.__conv_settings()

  def __set_input_con(self,__input):
    self.__input_con = __input

  def __get_input_con(self):
    return self.__input_con

  def convulate(self):
    print('Transforming all the images...')

    # self.samples = len(self.data['images'])

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
    kernel_size = [6,6,5] #6x6, 6x6, 5x5
    stride_size = [2,2,1] 
    polling_size = [0,2,2]# where zero is, there is no polling
    sample = self.data['images'][0]
    convs = {} 
    
    for l in range(len(maps)):
      if l == 0:
        __input = sample.reshape((300,300,1))
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
    df.to_csv(self.root + '/csv/process_' + str(self.process) + '/impa_samples.csv')
    df2.to_csv(self.root + '/csv/process_' + str(self.process) + '/impa_labels.csv')    
  
    print('CSV file has been saved!')

impa = IMPA()
dataset = impa.data()
impa.save_some_samples()
samples = len(dataset['images'])
cnn = CNN(data = dataset, process = 2, samples = samples)
cnn.convulate()