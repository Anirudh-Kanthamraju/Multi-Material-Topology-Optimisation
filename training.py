# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:43:27 2021

@author: anika
"""
from Network import Generator, Discriminator

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model

from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import keras.losses

import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
import csv
import os


# Values of some blobal variables 

training_ratio =0.8
image_shape_HR = [40,80,3]
image_shape_LR= [20,40,3]
downscale_factor_axis1= image_shape_HR[1]/image_shape_LR[1]
downscale_factor_axis2= image_shape_HR[0]/image_shape_LR[0]

if downscale_factor_axis1==downscale_factor_axis2:
    downscale_factor = downscale_factor_axis1
    print("Image Shapes Compatable")
    print ("Downscale Factor = ", downscale_factor )
else:
    print("Image Shapes Incompatable")
 
##############################################################################vgg
# Loading the data
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))

    return directories
    
def load_data_from_dirs(dirs, ext):          
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = io.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                    #print(file_names,"\n")
                count = count + 1
                
    
    return files
          
def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


files_HR = load_data("./data_HR", ".jpg")
files_LR = load_data("./data_LR", ".jpg")

files_HR_train = files_HR[:int(len(files_HR)*training_ratio)]
files_HR_test = files_HR[int(len(files_HR)*training_ratio):len(files_HR)]
files_LR_train = files_LR[:int(len(files_LR)*training_ratio)]
files_LR_test = files_LR[int(len(files_LR)*training_ratio):len(files_HR)]
    

#        The way the data is being loaded is a mystry to me. But the fact that, 
#        both the LR and HR images have the same name simplified the loading 
#        and there is no need for writing a sorting function. 
print("data loaded")
##############################################################################




##############################################################################
# Processing and Normalizing the images 
def array_images(images):
    image_array = array(images)
    return image_array

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 

images_hr_test =array_images(files_HR_test) 
images_hr_train =array_images(files_HR_train)
images_lr_test =array_images(files_LR_test)
images_lr_train =array_images(files_LR_train)

images_hr_test  = normalize(images_hr_test)
images_lr_test  = normalize(images_lr_test)
images_hr_train = normalize(images_hr_train)
images_lr_train = normalize(images_lr_train)




print("Data processed and Normalized")
##############################################################################






##############################################################################
def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape_HR)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))




def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


##############################################################################






##############################################################################

# Let the trains begin xD  the train module


def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    rand_nums = np.random.randint(0, images_hr_test.shape[0], size=examples)
    image_batch_hr = denormalize(images_hr_test[rand_nums])
    image_batch_lr = images_lr_test[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    
    name1 = 'output/gen_outputepoch='
    name2 = str(epoch)
    extension = '.jpg'
    name = name1+name2+extension
    plt.savefig(name)

def train(epochs, batch_size):
    data= []
    batch_count = int(images_hr_train.shape[0] / batch_size)
    shape = image_shape_LR
 
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape_HR).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    

    gan = get_gan_network(discriminator, shape, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            
            rand_nums = np.random.randint(0, images_hr_train.shape[0], size=batch_size)
            
            image_batch_hr = images_hr_train[rand_nums]
            image_batch_lr = images_lr_train[rand_nums]

            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
          
            rand_nums = np.random.randint(0, images_hr_train.shape[0], size=batch_size)
            image_batch_hr = images_hr_train[rand_nums]
            image_batch_lr = images_lr_train[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            print("Loss HR , Loss LR, Loss GAN")
            print(d_loss_real, d_loss_fake, loss_gan) 
            data= data + [(e,d_loss_real, d_loss_fake, loss_gan)]
        if e == 1 or e % 10 == 0:
            plot_generated_images(e, generator)
        if e % 100 == 0:
            generator.save('./output/gen_model%d.h5' % e)
            discriminator.save('./output/dis_model%d.h5' % e)
            gan.save('./output/gan_model%d.h5' % e)
        with open('LossData.csv','w') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['epoch','Loss HR' , 'Loss LR', 'Loss GAN'])
            for row in data:
                csv_out.writerow(row)

train (5000, 16)

