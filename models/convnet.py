import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Permute, Conv2D, MaxPooling2D,
                                     Dropout, Flatten, Dense)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ConvNet(object):
    def __init__(self, img_rows, img_cols, img_chans, n_classes,
                 optimizer=Adam, loss='categorical_crossentropy',
                 metrics=['acc'], learning_rate=3e-04, model_path = 'assets/nn-model.hdf5'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_chans = img_chans
        self.n_classes = n_classes
        self.model_path = model_path

        self.optimizer = Adam(lr=learning_rate)
        self.loss = loss
        self.metrics = metrics

        self.model = None

    def build_model(self):
        input_layer = Input(shape=(self.img_rows, self.img_cols, self.img_chans))

        # handle image dimensions ordering
        if tf.keras.backend.image_data_format() == 'channels_first':
            latent = Permute((3, 1, 2))(input_layer)
        else:
            latent = input_layer

        # define the network architecture
        latent = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(latent)
        latent = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(latent)
        latent = MaxPooling2D(pool_size=(2, 2))(latent)
        latent = Dropout(rate=0.25)(latent)

        latent = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(latent)
        latent = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(latent)
        latent = MaxPooling2D(pool_size=(2, 2))(latent)
        latent = Dropout(rate=0.25)(latent)

        latent = Flatten()(latent)
        latent = Dense(units=512, activation='relu')(latent)
        latent = Dropout(rate=0.5)(latent)

        output_layer = Dense(units=self.n_classes, activation='softmax')(latent)

        self.model = Model(inputs=input_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                        metrics=self.metrics)

    def maybe_train(self, data_train, data_valid, batch_size, epochs, callbacks=None):

        path_model = self.model_path
        dir_assets = os.path.dirname(path_model)

        if os.path.exists(path_model):
            print(f'🔁 Loading trained model from {path_model}.')
            self.model = load_model(path_model, compile=False)
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        else:
            print(f'🚀 No checkpoint found on {path_model}. Training from scratch.')
            self.build_model()
            x_train, y_train = data_train
            self.model.fit(x_train, y_train, validation_data=data_valid,
                           batch_size=batch_size, callbacks=callbacks, epochs=epochs)
            print(f'💾 Saving trained model to {path_model}.')
            if not os.path.isdir(dir_assets):
                os.makedirs(dir_assets)
            self.model.save(path_model)

    def evaluate(self, x, y):
        if self.model:
            # 모델 예측
            y_pred_probs = self.model.predict(x)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y, axis=1)

            # 모델 평가 (loss, accuracy)
            score = self.model.evaluate(x, y, verbose=0)

            # 정확도 및 기타 지표 계산
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            print('Evaluation Results:')
            print(' - Loss:      {:.4f}'.format(score[0]))
            print(' - Accuracy:  {:.2f}%'.format(100 * acc))
            print(' - Precision: {:.2f}%'.format(100 * precision))
            print(' - Recall:    {:.2f}%'.format(100 * recall))
            print(' - F1-Score:  {:.2f}%'.format(100 * f1))
        else:
            print('Missing model instance.')

    def predict(self, x):
        if self.model:
            return self.model.predict(x, verbose=1)
        else:
            print('Missing model instance.')
