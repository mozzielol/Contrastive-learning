import keras.datasets as Datasets
import numpy as np
import keras


class Dataloader(keras.utils.Sequence):
    def __init__(self, args, shuffle=True, subset='train'):
        self.args = args
        self.shuffle = shuffle
        self.height, self.width = 32, 32
        self.subset = subset
        self.load()

    def load(self):
        dataset_obj = getattr(Datasets, self.args.dataset)
        (X_train, y_train), (X_test, y_test) = dataset_obj.load_data()
        if self.args.flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        else:
            if self.args.dataset in ['mnist']:
                X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
                X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

            elif self.args.dataset in ['cifar10', 'cifar100']:
                X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
                X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

            elif self.args.dataset in ['omniglot']:
                X_train = X_train.reshape(X_train.shape[0], 105, 105, 1)
                X_test = X_test.reshape(X_test.shape[0], 105, 105, 1)

        if np.max(X_train) == 255.:
            print('Normalizing the training data ... ')
            X_train = X_train.astype('float32') / 255
            X_test = X_test.astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
        y_test = keras.utils.to_categorical(y_test, len(np.unique(y_test)))
        self.indexes = np.arange(y_train.shape[0])
        self.iteration = y_train.shape[0] // self.args.batch_size
        self.data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }
        return X_train, y_train, X_test, y_test

    def __len__(self):
        return int(np.ceil(len(self.data['X_train']) / float(self.args.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def sample(self):
        X = np.empty(
            (self.args.batch_size, self.height, self.width, 3),
            dtype=np.float32,
        )

        # indexes = self.indexes[
        #     index * self.args.batch_size: (index + 1) * self.args.batch_size
        # ]
        indexes = np.random.choice(self.indexes, size=self.args.batch_size, replace=False)
        batch_data = self.data['X_train'][indexes]

        # Add shuffle in order to avoid network recalling fixed order
        shuffle_a = np.arange(self.args.batch_size)
        shuffle_b = np.arange(self.args.batch_size)

        if self.subset == "train":
            np.random.shuffle(shuffle_a)
            np.random.shuffle(shuffle_b)
        if self.subset == "val":
            # Exclude randomness for evaluation
            np.random.seed(42)
            np.random.shuffle(shuffle_a)
            np.random.shuffle(shuffle_b)

        labels = np.zeros((self.args.batch_size, 2 * self.args.batch_size))

        for i, row in enumerate(batch_data):
            # T1-images between 0 -> batch_size - 1
            X[shuffle_a[i]] = batch_data[shuffle_a[i]]
            # T2-images between batch_size -> 2*batch_size - 1
            # label ab
            labels[shuffle_a[i], shuffle_a[i]] = 1
            # label ba
            labels[shuffle_a[i], shuffle_a[i] + self.args.batch_size] = 1


        # [None] is used to silence warning
        # https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
        return X, labels

