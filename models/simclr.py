from models.base_blocks import get_autoencoder, get_projector
from keras.models import Model
from models.self_attention import SelfAttention
from models.loss import add_contrastive_loss, cos_similarity
import keras.backend as K
import keras
from keras.layers import Lambda
import numpy as np


class SimCLR:
    def __init__(self, args):
        # build the model
        self.args = args
        self.encoder, _ = get_autoencoder()
        main_obj, background = SelfAttention(32)(self.encoder.output)
        self.projector = get_projector(K.int_shape(main_obj)[1:], args.feat_dims)
        main_obj = self.projector(main_obj)
        background = self.projector(background)
        #self.model = Model(self.encoder.input,
        #                   Lambda(lambda x: SoftmaxCosineSim(args.batch_size, args.feat_dims)(x))(keras.layers.concatenate([main_obj, background], axis=0)))
        # self.model = Model(self.encoder.input, Lambda(lambda x: add_contrastive_loss(x))(keras.layers.concatenate([main_obj, background], axis=0)))
        self.model = Model(self.encoder.input, Lambda(lambda x: cos_similarity(x))(
            keras.layers.concatenate([main_obj, background], axis=0)))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, loader):
        # 1. extract the features
        # self.model.fit(loader, epochs=self.args.epoch, verbose=self.args.verbose)
        self.model.fit(loader.data['X_train'],np.ones(loader.data['X_train'].shape[0]), epochs=self.args.epoch, verbose=self.args.verbose)
        # for _ in range(self.args.epoch):
        #     for _ in np.arange(loader.iteration):
        #         X_train, y_train = loader.sample()
        #         loss = self.model.train_on_batch(X_train, y_train)
        #         print('loss :', loss, end='\r')
