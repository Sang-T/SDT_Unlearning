import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Permute, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ConvNet_CIFAR(object):
    def __init__(self, img_rows, img_cols, img_chans, n_classes,
                 optimizer=Adam, loss='categorical_crossentropy',
                 metrics=['acc'], learning_rate=0.001, model_path='assets/nn-model.hdf5'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_chans = img_chans
        self.n_classes = n_classes
        self.model_path = model_path
        self.learning_rate = learning_rate

        self.optimizer = Adam(lr=learning_rate, decay=1e-06) # RMSprop
        self.loss = loss
        self.metrics = metrics
        self.model = None


    def build_model(self, learning_rate=None):
        weight_decay = 1e-04
        baseMapNum = 32
        input_layer = Input(shape=(self.img_rows, self.img_cols, self.img_chans))

        if tf.keras.backend.image_data_format() == 'channels_first':
            x = Permute((3, 1, 2))(input_layer)
        else:
            x = input_layer

        if learning_rate is None:
            learning_rate = self.learning_rate

        # Block 1
        x = Conv2D(baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(input_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x) # 0.25

        # Block 2
        x = Conv2D(2 * baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(2 * baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x) # 0.25

        # Block 3
        x = Conv2D(4 * baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(4 * baseMapNum, (3, 3), padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x) # 0.25

        # Classification head
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x) # 0.25
        output_layer = Dense(self.n_classes, activation='softmax')(x)

        # Build & compile model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def maybe_train(self, data_train, data_valid, batch_size, epochs, callbacks=None, use_datagen=False):
        path_model = self.model_path
        dir_assets = os.path.dirname(path_model)

        if os.path.exists(path_model):
            try:
                print(f'🔁 Trying to load full model from {path_model}.')
                self.model = load_model(path_model, compile=False)
                self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            except:
                print(f'🧩 Detected weights-only file. Building model and loading weights from {path_model}.')
                self.build_model()
                self.model.load_weights(path_model)
        else:
            print(f'🚀 No checkpoint found on {path_model}. Training from scratch.')
            self.build_model()
            x_train, y_train = data_train
            x_valid, y_valid = data_valid

            if use_datagen:
                print("🔄 Using ImageDataGenerator for data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=8, # cifar는 15
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=False, # cifar10은 true
                    vertical_flip=False
                )
                datagen.fit(x_train)

                for lr, stage_epochs in zip([0.0005, 0.001, 0.01], [75, 25, 25]):
                    print(f"📘 Training with learning rate: {lr}")
                    weights_backup = self.model.get_weights() if self.model else None
                    self.optimizer = Adam(lr=lr, decay=1e-6, amsgrad = True) # cifar 10에서는 RMSprop
                    self.build_model(learning_rate=lr)
                    if weights_backup is not None:
                        self.model.set_weights(weights_backup)
                    self.model.fit(
                        datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=stage_epochs,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=1
                    )

                # self.model.fit(
                #     datagen.flow(x_train, y_train, batch_size=batch_size),
                #     steps_per_epoch=x_train.shape[0] // batch_size,
                #     validation_data=(x_valid, y_valid),
                #     epochs=epochs,
                #     callbacks=callbacks,
                #     verbose=1
                # )
            else:
                self.model.fit(
                    x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )

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

    def predict(self, x, batch_size = 64):
        if self.model:
            return self.model.predict(x, batch_size = batch_size, verbose=1)
        else:
            print('Missing model instance.')
