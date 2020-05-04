import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import tanh
import glob
import numpy as np
import datetime

class Pix2PixModel:
    """
    This class implements the pix2pix model in keras
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf

    Based on my previous paper: Animated Image Colorization with Pix2Pix (Balaji, Powers, Pari)
    """
    def __init__(self, original_shape=(256,256,3), translated_shape=(256,256,3)):
        self.original_shape = original_shape
        self.translated_shape = translated_shape
        
        print("Setting up Discriminator")
        self.disc_model, discriminator = self.create_discriminator()
        _, patch_height, patch_width, _ = discriminator.shape
        self.patch_height = patch_height
        self.patch_width = patch_width

        print("Setting up Generator")
        self.gen_model = self.create_generator()
        self.disc_model.trainable = False

        self.GANModel = self.create_gan(self.gen_model, self.disc_model)

        # Used for suppressing warnings
        self.disc_model.trainable = True


    
    def create_gan(self, gen_model, disc_model, image_shape=(256,256,3)):
        # Freeze the discriminator weights while training the generator
        disc_model.trainable = False

        real_orignal_input = Input(self.original_shape)
        generated_translated_image = gen_model(real_orignal_input)
        disc_output = disc_model([real_orignal_input, generated_translated_image])
        # ful model; input: source image; output: generated_translated_image, classification
        model = Model(real_orignal_input, [disc_output, generated_translated_image])

        opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
        model.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100], optimizer=opt)

        return model


    def create_generator(self):
        real_orignal_input = Input(self.original_shape)
        encoding_layers = []

        e = self.create_encoding_layer(real_orignal_input, 64, batch_norm=False)
        encoding_layers.append(e)
        for num_filters in [128,256,512,512,512,512]:
            e = self.create_encoding_layer(e, num_filters)
            encoding_layers.append(e)
        
        bottleneck = self.create_convolution(e, 512)
        d = ReLU()(bottleneck)

        num_filters_list = [512,512,512,512,256,128,64]
        num_decoding_layers = len(num_filters_list)
        for idx in range(num_decoding_layers):
            num_filters = num_filters_list[idx]
            e_skip_layer = encoding_layers[num_decoding_layers - idx - 1]
            d = self.create_decoding_layer(d, e_skip_layer, num_filters, dropout=num_filters > 2)
        
        output = self.create_deconvolution(d, 3)
        output = Activation('tanh')(output)

        # define model
        model = Model(inputs=real_orignal_input, outputs=output)
        return model
        

    def create_encoding_layer(self, input, num_filters, batch_norm = True):
        l = self.create_convolution(input, num_filters)
        if batch_norm:
            l = BatchNormalization(axis = 3)(l)
        l = LeakyReLU(alpha=0.2)(l)
        return l

    def create_decoding_layer(self, input, e_skip, num_filters, dropout=True):
        l = self.create_deconvolution(input, num_filters)
        l = BatchNormalization(axis = 3)(l)
        if dropout:
            l = Dropout(0.5)(l)
        l = Concatenate()([l, e_skip])
        l = ReLU()(l)
        return l

    def create_discriminator(self):
        predict_translated_input = Input(self.translated_shape)
        real_orignal_input = Input(self.original_shape)

        concat = concatenate([predict_translated_input, real_orignal_input])

        # 0. 64 Filters; ret -> 128 x 128 patch grid
        d = self.create_convolution(concat, 64)
        d = LeakyReLU(alpha=0.2)(d)

        # 1. 128 Filters; ret -> 64 x 64 patch grid
        d = self.create_convolution(d, 128)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # 2. 256 Filters; ret ->  32 x 32 patch grid
        d = self.create_convolution(d, 256)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # 3. 512 Filters; ret ->  16 x 16 patch grid
        d = self.create_convolution(d, 512)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # 4. 512 Filters; ret -> 16 x 16 patch grid
        d = self.create_convolution(d, 512, strides=(1,1))
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        # 5. Patch Output; 1 Filter; 16 x 16 patch grid
        d = self.create_convolution(d, 1, strides=(1,1))
        output = Activation('sigmoid')(d)
        
        # define model
        model = Model([predict_translated_input, real_orignal_input], outputs=output)

        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model, output


    def create_convolution(self, input, num_filters, strides=(2,2)):
        init = kernel_initializer=tf.random_normal_initializer(0, 0.02)
        return Conv2D(num_filters, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init)(input)
    
    def create_deconvolution(self, input, num_filters, activation=None):
        return Conv2DTranspose(num_filters, (4, 4), padding='same', strides=(2, 2), activation=activation)(input)


    def train_on_batch(self, original_im_batch, translated_im_batch, batch_size):
        real_target = np.ones((batch_size, self.patch_height, self.patch_width, 1))
        fake_target = np.zeros((batch_size, self.patch_height, self.patch_width, 1))

        print("og im batch: {}".format(np.array(original_im_batch).shape))
        print("tr im batch: {}".format(np.array(translated_im_batch).shape))

        fake_color = self.gen_model.predict(np.array(original_im_batch))
        disc_fake_loss = self.disc_model.train_on_batch([fake_color, original_im_batch], fake_target)
        disc_real_loss = self.disc_model.train_on_batch([translated_im_batch, original_im_batch], real_target)
        disc_loss = 0.5 * np.add(disc_fake_loss, disc_real_loss)

        gan_loss = self.GANModel.train_on_batch([original_im_batch], [real_target, translated_im_batch])

        return gan_loss, disc_loss

    def eval_on_batch(self, original_im_batch, translated_im_batch, batch_size):
        real_target = np.ones((batch_size, self.patch_height, self.patch_width, 1))
        fake_target = np.zeros((batch_size, self.patch_height, self.patch_width, 1))

        fake_color = self.gen_model.predict(original_im_batch)
        disc_fake_loss = self.disc_model.evaluate([fake_color, original_im_batch], fake_target)
        disc_real_loss = self.disc_model.evaluate([translated_im_batch, original_im_batch], real_target)
        disc_loss = 0.5 * np.add(disc_fake_loss, disc_real_loss)

        gan_loss = self.GANModel.evaluate([original_im_batch], [real_target, translated_im_batch])