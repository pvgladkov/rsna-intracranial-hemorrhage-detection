import keras
from rsna.data import DataGenerator
from rsna.loss import weighted_loss
from numpy import floor
from rsna.gem import GeM2D
from keras.layers.pooling import GlobalAveragePooling2D


class PredictionCheckpoint(keras.callbacks.Callback):

    def __init__(self, test_df, valid_df, test_images_dir, valid_images_dir,
                 batch_size=32, input_size=(224, 224, 3)):
        super(PredictionCheckpoint, self).__init__()
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size

    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []

    def on_epoch_end(self, batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, self.test_images_dir, None, self.batch_size, self.input_size),
                verbose=2)[:len(self.test_df)])


class Model:

    def __init__(self, engine, input_dims, train_images_dir, test_images_dir, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self.train_images_dir = train_images_dir
        self.test_images_dir = test_images_dir
        self._build()

    def _build(self):
        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims)

        x = GlobalAveragePooling2D(padding='valid')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = keras.models.Model(inputs=engine.input, outputs=out)

        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])
        self.model.summary()

    def fit_and_predict(self, train_df, valid_df, test_df):
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims,
                                            test_images_dir=self.test_images_dir, valid_images_dir=None)
        # checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                self.train_images_dir,
                train_df,
                self.batch_size,
                self.input_dims,
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
            callbacks=[pred_history, scheduler]
        )

        return pred_history

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)