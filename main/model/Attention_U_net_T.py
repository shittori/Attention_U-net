class attention_unet():
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.df = 64
        self.uf = 64

    def build_unet(self):
        def conv2d(layer_input, filters, dropout_rate=0, bn=False):
            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)

            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)

            if dropout_rate:
                d = layers.Dropout(dropout_rate)(d)

            return d

        def deconv2d(layer_input, filters, bn=False):
            u = layers.UpSampling2D((2, 2))(layer_input)
            u = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(u)
            if bn:
                u = layers.BatchNormalization()(u)
            u = layers.Activation('relu')(u)

            return u

        def attention_block(F_g, F_l, F_int, bn=False):
            g = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            if bn:
                g = layers.BatchNormalization()(g)
            x = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            if bn:
                x = layers.BatchNormalization()(x)
            #       print(g.shape)
            #       print(x.shape)
            psi = layers.Add()([g, x])
            psi = layers.Activation('relu')(psi)

            psi = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)

            if bn:
                psi = layers.BatchNormalization()(psi)
            psi = layers.Activation('sigmoid')(psi)

            return layers.Multiply()([F_l, psi])

        inputs = layers.Input(shape=self.img_shape)

        conv1 = conv2d(inputs, self.df)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)

        conv2 = conv2d(pool1, self.df * 2, bn=True)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        conv3 = conv2d(pool2, self.df * 4, bn=True)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)

        conv4 = conv2d(pool3, self.df * 8, dropout_rate=0.5, bn=True)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = conv2d(pool4, self.df * 16, dropout_rate=0.5, bn=True)

        up6 = deconv2d(conv5, self.uf * 8, bn=True)
        conv6 = attention_block(up6, conv4, self.uf * 8, bn=True)
        up6 = layers.Concatenate()([up6, conv6])
        conv6 = conv2d(up6, self.uf * 8)

        up7 = deconv2d(conv6, self.uf * 4, bn=True)
        conv7 = attention_block(up7, conv3, self.uf * 4, bn=True)
        up7 = layers.Concatenate()([up7, conv7])
        conv7 = conv2d(up7, self.uf * 4)

        up8 = deconv2d(conv7, self.uf * 2, bn=True)
        conv8 = attention_block(up8, conv2, self.uf * 2, bn=True)
        up8 = layers.Concatenate()([up8, conv8])
        conv8 = conv2d(up8, self.uf * 2)

        up9 = deconv2d(conv8, self.uf, bn=True)
        conv9 = attention_block(up9, conv1, self.uf, bn=True)
        up9 = layers.Concatenate()([up9, conv9])
        conv9 = conv2d(up9, self.uf)

        outputs = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        return model