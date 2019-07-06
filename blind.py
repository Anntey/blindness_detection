
import pandas as pd
import numpy as np
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine.input_layer import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50

# loading data
input_path = "./input/"

train_df = pd.read_csv(input_path + "train.csv")
test_df = pd.read_csv(input_path + "test.csv")
test_df = pd.read_csv(input_path + "sample_submission.csv")

train_df["id_code"] = train_df["id_code"].apply(lambda x: x + ".png")
test_df["id_code"] = test_df["id_code"].apply(lambda x: x + ".png")
train_df["diagnosis"] = train_df["diagnosis"].astype(str) # keras requires str

labels_list = ["0", "1", "2", "3", "4"]

train_df["diagnosis"].value_counts()

# generators and data augmentation
train_datagen = ImageDataGenerator(
        rescale = 1./255, 
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        validation_split = 0.2,
)

test_datagen = ImageDataGenerator(
        rescale = 1./255
)

train_gen = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = input_path + "train_images",
        x_col = "id_code",
        y_col = "diagnosis",
        batch_size = 32,
        shuffle = True,
        class_mode = "categorical",
        classes = labels_list,
        target_size = (64, 64),
        subset = "training"
)


val_gen = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory= input_path + "train_images",
        x_col = "id_code",
        y_col = "diagnosis",
        batch_size = 32,
        shuffle = True,
        class_mode = "categorical",
        classes = labels_list,
        target_size = (64, 64),
        subset = "validation"
)    

test_gen = test_datagen.flow_from_dataframe(  
        dataframe = test_df,
        directory = input_path + "test_images",    
        x_col = "id_code",
        target_size = (64, 64),
        batch_size = 1,
        shuffle = False,
        class_mode = None
)

# specifying model (transfer learning)
inp = Input(shape = (64, 64, 3))

base_model = ResNet50(include_top = False, weights = None, input_tensor = inp)
base_model.load_weights(input_path + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

base_model.trainable = False # freeze weights

x = GlobalAveragePooling2D()(base_model.output) # adding top layers
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(5, activation = "softmax")(x)

model = Model(inp, x)

model.compile(
        optimizer = optimizers.Adam(lr = 1e-4, decay = 1e-6),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
)

# fitting model
callbacks_list = [
        EarlyStopping(
                monitor = "val_loss", 
                mode = "min",
                patience = 6
        ),
        ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.5,
                patience = 4,
                verbose = 1,
                mode = "min",
        ),
]

history = model.fit_generator(
        train_gen, 
        validation_data = val_gen,
        epochs = 18,
        steps_per_epoch = 92,
        validation_steps = 23,
        callbacks = callbacks_list,
        #max_queue_size = 16,
        #workers = 2,
        #use_multiprocessing = True,
        #verbose = 2
)

# evaluation
history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
history_df[["acc", "val_acc"]].plot()

# prediction
preds = model.predict_generator(test_gen, steps = len(test_gen))
preds = np.argmax(preds, axis = 1)
subm_df = pd.DataFrame({"id_code": test_gen.filenames, "diagnosis": preds})

subm_df["id_code"] = subm_df["id_code"].map(lambda x: str(x)[:-4]) # remove ".png"
subm_df.to_csv("subm.csv", index = False)
