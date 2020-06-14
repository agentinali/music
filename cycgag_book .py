
import sys
import os
from PIL import Image

import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate,Conv2D
from keras.layers import UpSampling2D
from keras.layers.core import Activation
# from keras_contrib.layers.normalization import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt
    
class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        
        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        # self.save_model()
#         self.summary()

    def model(self):
        input_layer = Input(self.SHAPE)

        up_layer_1 = Conv2D(64, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

        up_layer_2 = Conv2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        norm_layer_1 = InstanceNormalization()(up_layer_2)

        up_layer_3 = Conv2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_1)
        norm_layer_2 = InstanceNormalization()(up_layer_3)

        up_layer_4 = Conv2D(64*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_2)
        norm_layer_3 =InstanceNormalization()(up_layer_4)

        output_layer = Conv2D(1, kernel_size=4, strides=1, padding='same')(norm_layer_3)
        output_layer_1 = Flatten()(output_layer)
        output_layer_2 = Dense(1, activation='sigmoid')(output_layer_1)
        
        return Model(input_layer,output_layer_2)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='./data/Discriminator_Model.png')

class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (width,height,channels)

        self.Generator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER,metrics=['accuracy'])

        # self.save_model()
#         self.summary()

    def model(self):
        input_layer = Input(shape=self.SHAPE)
        
        down_1 = Conv2D(64  , kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
        norm_1 = InstanceNormalization()(down_1)

        down_2 = Conv2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_1)
        norm_2 = InstanceNormalization()(down_2)

        down_3 = Conv2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 = InstanceNormalization()(down_3)

        down_4 = Conv2D(64*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 = InstanceNormalization()(down_4)


        upsample_1 = UpSampling2D()(norm_4)
        up_conv_1 = Conv2D(64*4, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_1)
        norm_up_1 = InstanceNormalization()(up_conv_1)
        add_skip_1 = Concatenate()([norm_up_1,norm_3])

        upsample_2 = UpSampling2D()(add_skip_1)
        up_conv_2 = Conv2D(64*2, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_2)
        norm_up_2 = InstanceNormalization()(up_conv_2)
        add_skip_2 = Concatenate()([norm_up_2,norm_2])

        upsample_3 = UpSampling2D()(add_skip_2)
        up_conv_3 = Conv2D(64, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_3)
        norm_up_3 = InstanceNormalization()(up_conv_3)
        add_skip_3 = Concatenate()([norm_up_3,norm_1])

        last_upsample = UpSampling2D()(add_skip_3)
        output_layer = Conv2D(self.C, kernel_size=4, strides=1, padding='same',activation='tanh')(last_upsample)
        
        return Model(input_layer,output_layer)

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='./datax/Generator_Model.png')

class GAN(object):
    def __init__(self, model_inputs=[],model_outputs=[],lambda_cycle=1.0,lambda_id=1.0):
        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)

        self.inputs = model_inputs
        self.outputs = model_outputs
        self.gan_model = Model(self.inputs,self.outputs)
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        self.gan_model.compile(loss=['mse', 'mse',
                                     'mae', 'mae',
                                     'mae', 'mae'],
                       loss_weights=[ 1, 1,
                                      lambda_cycle, lambda_cycle,
                                      lambda_id, lambda_id ],
                          optimizer=self.OPTIMIZER)
        # self.save_model()
#         self.summary()

    def model(self):
        model = Model()
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='./datax/GAN_Model.png')

class Trainer:
    def __init__(self, height = 64, width = 64, epochs = 50000, batch = 32, checkpoint = 50, train_data_path_A = '',train_data_path_B = '',test_data_path_A='',test_data_path_B='',lambda_cycle=10.0,lambda_id=1.0):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.RESIZE_HEIGHT = height
        self.RESIZE_WIDTH = width
        self.CHECKPOINT = checkpoint

        self.X_train_A, self.H_A, self.W_A, self.C_A = self.load_data(train_data_path_A)
        self.X_train_B, self.H_B, self.W_B, self.C_B  = self.load_data(train_data_path_B)
        self.X_test_A, self.H_A_test, self.W_A_test, self.C_A_test = self.load_data(test_data_path_A)
        self.X_test_B, self.H_B_test, self.W_B_test, self.C_B_test  = self.load_data(test_data_path_B)
       
        self.generator_A_to_B = Generator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.generator_B_to_A = Generator(height=self.H_B, width=self.W_B, channels=self.C_B)

        self.orig_A = Input(shape=(self.W_A, self.H_A, self.C_A))
        self.orig_B = Input(shape=(self.W_B, self.H_B, self.C_B))

        self.fake_B = self.generator_A_to_B.Generator(self.orig_A)
        self.fake_A = self.generator_B_to_A.Generator(self.orig_B)
        self.reconstructed_A = self.generator_B_to_A.Generator(self.fake_B)
        self.reconstructed_B = self.generator_A_to_B.Generator(self.fake_A)
        self.id_A = self.generator_B_to_A.Generator(self.orig_A)
        self.id_B = self.generator_A_to_B.Generator(self.orig_B)


        self.discriminator_A = Discriminator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.discriminator_B = Discriminator(height=self.H_B, width=self.W_B, channels=self.C_B)
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.valid_A = self.discriminator_A.Discriminator(self.fake_A)
        self.valid_B = self.discriminator_B.Discriminator(self.fake_B)

        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid_A, self.valid_B,self.reconstructed_A,self.reconstructed_B,self.id_A, self.id_B]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs,lambda_cycle=lambda_cycle,lambda_id=lambda_id)
        
      
        

    def load_data(self,data_path,amount_of_data = 1.0):
        listOFFiles = self.grabListOfFiles(data_path,extension="png")
     
        X_train = np.array(self.grabArrayOfImages(listOFFiles,gray=True))
#         X_train = np.array(self.grabArrayOfImages(listOFFiles))
        if len(np.shape(X_train[0]))==2:
            height, width = np.shape(X_train[0])
            channels=1
        else:    
            height, width, channels = np.shape(X_train[0])
        X_train = X_train[:int(amount_of_data*float(len(X_train)))]
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
       
        X_train = np.expand_dims(X_train, axis=3)
       
        return X_train, height, width, channels

    def grabListOfFiles(self,startingDirectory,extension=".webp"):
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))
        return listOfFiles

    def grabArrayOfImages(self,listOfFiles,gray=False):
        imageArr = []
        for f in listOfFiles:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")
            im = im.resize((self.RESIZE_WIDTH,self.RESIZE_HEIGHT))
            imData = np.asarray(im)
            imageArr.append(imData)
        return imageArr


    def train(self):
        if os.path.isfile("./datax/weights/d_A_epoch.h5"):
            print("load_weights......... ")
            self.discriminator_A.Discriminator.load_weights("./datax/weights/d_A_epoch.h5")
            self.discriminator_B.Discriminator.load_weights("./datax/weights/d_B_epoch.h5")
            self.generator_A_to_B.Generator.load_weights("./datax/weights/g_AB_epoch.h5")
            self.generator_B_to_A.Generator.load_weights("./datax/weights/g_BA_epoch.h5")
                        
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)
        
            while min(len(X_train_A_temp),len(X_train_B_temp))>self.BATCH:
                # Keep track of Batches
                b=b+1

                # Train Discriminator
                # Grab Real Images for this training batch

                count_real_images = int(self.BATCH)
                starting_indexs = randint(0, (min(len(X_train_A_temp),len(X_train_B_temp))-count_real_images))
                real_images_raw_A = X_train_A_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                real_images_raw_B = X_train_B_temp[ starting_indexs : (starting_indexs + count_real_images) ]

                # Delete the images used until we have none left
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_indexs,(starting_indexs + count_real_images)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_indexs,(starting_indexs + count_real_images)),0)
                batch_A = real_images_raw_A.reshape( count_real_images, self.W_A, self.H_A, self.C_A )
                batch_B = real_images_raw_B.reshape( count_real_images, self.W_B, self.H_B, self.C_B )

                self.discriminator_A.Discriminator.trainable = True
                self.discriminator_B.Discriminator.trainable = True
                x_batch_A = batch_A
                x_batch_B = batch_B
                y_batch_A = np.ones([count_real_images,1])
                y_batch_B = np.ones([count_real_images,1])
                # Now, train the discriminator with this batch of reals
                discriminator_loss_A_real = self.discriminator_A.Discriminator.train_on_batch(x_batch_A,y_batch_A)[0]
                discriminator_loss_B_real = self.discriminator_B.Discriminator.train_on_batch(x_batch_B,y_batch_B)[0]

                x_batch_B = self.generator_A_to_B.Generator.predict(batch_A)
                x_batch_A = self.generator_B_to_A.Generator.predict(batch_B)
                y_batch_A = np.zeros([self.BATCH,1])
                y_batch_B = np.zeros([self.BATCH,1])
                # Now, train the discriminator with this batch of fakes
                discriminator_loss_A_fake = self.discriminator_A.Discriminator.train_on_batch(x_batch_A,y_batch_A)[0]
                discriminator_loss_B_fake = self.discriminator_B.Discriminator.train_on_batch(x_batch_B,y_batch_B)[0]    

                self.discriminator_A.Discriminator.trainable = False
                self.discriminator_B.Discriminator.trainable = False

                discriminator_loss_A = 0.5*(discriminator_loss_A_real+discriminator_loss_A_fake)
            
                # In practice, flipping the label when training the generator improves convergence
#                 if self.flipCoin(chance=0.4):
#                     y_generated_labels = np.ones([self.BATCH,1])
#                 else:
#                     y_generated_labels = np.zeros([self.BATCH,1])
                    
#                 y_generated_labels = np.zeros([self.BATCH,1]) 
                y_generated_labels = np.ones([self.BATCH,1])
#                 generator_loss = self.gan.gan_model.train_on_batch([x_batch_A, x_batch_B],
                generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B],
                                                        [y_generated_labels, y_generated_labels,
                                                        x_batch_A, x_batch_B,
                                                        x_batch_A, x_batch_B])    

#                 print ('Batch: '+str(int(b))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)
                    print ('Batch: '+str(int(b))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                
            print ('Epoch: '+str(int(e))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
            os.makedirs('./datax/weights/' , exist_ok=True)
            self.discriminator_A.Discriminator.save_weights("./datax/weights/d_A_epoch.h5")
            self.discriminator_B.Discriminator.save_weights("./datax/weights/d_B_epoch.h5")
            self.generator_A_to_B.Generator.save_weights("./datax/weights/g_AB_epoch.h5")
            self.generator_B_to_A.Generator.save_weights("./datax/weights/g_BA_epoch.h5")
#             self.discriminator_A.Discriminator.save_weights("./data/weights/d_A_epoch%d.h5" % ( e))
#             self.discriminator_B.Discriminator.save_weights("./data/weights/d_B_epoch%d.h5" % ( e))
#             self.generator_A_to_B.Generator.save_weights("./data/weights/g_AB_epoch%d.h5" % ( e))
#             self.generator_B_to_A.Generator.save_weights("./data/weights/g_BA_epoch%d.h5" % ( e))
                        
#             if e % self.CHECKPOINT == 0 :
#                 self.plot_checkpoint(e)
        return

    def flipCoin(self,chance=0.9):
        return np.random.binomial(1, chance)

    def plot_checkpoint(self,b):
        orig_filename = "./datax/batch_check_"+str(b)+"_original.png"

        image_A = self.X_test_A[15]
        image_A = np.reshape(image_A, [self.W_A_test,self.H_A_test,self.C_A_test])
#         print("Image_A shape: " +str(np.shape(image_A)))
        fake_B = self.generator_A_to_B.Generator.predict(image_A.reshape(1, self.W_A, self.H_A, self.C_A ))
        fake_B = np.reshape(fake_B, [self.W_A_test,self.H_A_test,self.C_A_test])
#         print("fake_B shape: " +str(np.shape(fake_B)))
        reconstructed_A = self.generator_B_to_A.Generator.predict(fake_B.reshape(1, self.W_A, self.H_A, self.C_A ))
        reconstructed_A = np.reshape(reconstructed_A, [self.W_A_test,self.H_A_test,self.C_A_test])
#         print("reconstructed_A shape: " +str(np.shape(reconstructed_A)))
        # from IPython import embed; embed()

        checkpointA_images = np.array([image_A, fake_B, reconstructed_A])

        # Rescale images 0 - 1
        checkpointA_images = 0.5 * checkpointA_images + 0.5

        #---------------------------------------------------------
        orig_filename = "./datax/batch_checkB_"+str(b)+"_original.png"

        image_B = self.X_test_B[15]
        image_B = np.reshape(image_B, [self.W_B_test,self.H_B_test,self.C_B_test])
#         print("Image_B shape: " +str(np.shape(image_B)))
        fake_A = self.generator_B_to_A.Generator.predict(image_B.reshape(1, self.W_B, self.H_B, self.C_B ))
        fake_A = np.reshape(fake_A, [self.W_B_test,self.H_B_test,self.C_B_test])
#         print("fake_A shape: " +str(np.shape(fake_A)))
        reconstructed_B = self.generator_A_to_B.Generator.predict(fake_A.reshape(1, self.W_B, self.H_B, self.C_B ))
        reconstructed_B = np.reshape(reconstructed_B, [self.W_B_test,self.H_B_test,self.C_B_test])
#         print("reconstructed_B shape: " +str(np.shape(reconstructed_B)))
        # from IPython import embed; embed()

        checkpointB_images = np.array([image_B, fake_A, reconstructed_B])

        # Rescale images 0 - 1
        checkpointB_images = 0.5 * checkpointB_images + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axes = plt.subplots(2, 3)
        for i in range(2):
            if i==0:
                checkpoint_images= checkpointA_images
            if i==1:
                checkpoint_images= checkpointB_images    
            for k in range(3):
                image = checkpoint_images[k]
                image = np.reshape(image, [self.H_B_test,self.W_B_test,self.C_B_test])
                axes[i,k].imshow(image.squeeze(),plt.cm.gray)
                axes[i,k].set_title(titles[k])
                axes[i,k].axis('off')
        fig.savefig("./datax/batch_check_"+str(b)+".png")
        plt.close('all')

        return


if __name__=='__main__':
# Command Line Argument Method
    HEIGHT  = 32
    WIDTH   = 32
    CHANNEL = 1
    EPOCHS = 5
    BATCH = 1
    CHECKPOINT =500

    TRAIN_PATH_A = "./datax/trainA/"
    TRAIN_PATH_B = "./datax/trainB/"
    TEST_PATH_A = "./datax/testA/"
    TEST_PATH_B = "./datax/testB/"

    trainer = Trainer(height=HEIGHT,width=WIDTH,epochs =EPOCHS, batch=BATCH, checkpoint=CHECKPOINT, train_data_path_A=TRAIN_PATH_A, train_data_path_B=TRAIN_PATH_B, test_data_path_A=TEST_PATH_A, test_data_path_B=TEST_PATH_B, lambda_cycle=10.0, lambda_id=1.0)
    trainer.train()

#
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 3)
#
# data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
#
# data.plot.bar(ax=axes[1,1], color='b', alpha = 0.5)
# data.plot.bar(ax=axes[1,0], color='K', alpha = 0.5)
#
# data.plot.barh(ax=axes[0,1], color='k', alpha=0.5)
# plt.show()




