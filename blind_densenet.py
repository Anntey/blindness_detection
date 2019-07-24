#############
# Libraries #
#############

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input
from keras.applications import DenseNet121
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

################
# Loading data #
################

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_df["diagnosis"].value_counts()

########################
# Preprocessing images #
########################

y_train = to_categorical(train_df["diagnosis"], num_classes = 5)

x_train = []

for i, img_id in enumerate(train_df["id_code"]):
    img = cv2.imread("./input/train_images/" + img_id + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 150/10), -4 , 128) # credit to Ben Graham
    x_train.append(img)

x_train = np.asarray(x_train, dtype = "float32")
x_train = x_train / 255.0 # normalize

x_test = []

for i, img_id in enumerate(test_df["id_code"]):
    img = cv2.imread("../input/aptos2019-blindness-detection/test_images/" + img_id + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 150/10), -4 , 128)
    x_test.append(img)
    if i % 500 == 0:
        print(i)

x_test = np.asarray(x_test, dtype = "float32")
x_test = x_test / 255.0

######################
# Visualizing images #
######################

img_eg = cv2.imread("./input/test_images/009c019a7309.png")  # visualizing random image
img_eg = cv2.cvtColor(img_eg, cv2.COLOR_BGR2RGB)
img_eg = cv2.resize(img_eg, (224, 224))

fig = plt.figure(figsize = (10, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img_eg)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(x_test[5])

##############
# Generators #
##############

train_datagen = ImageDataGenerator(
    zoom_range = 0.15,
    fill_mode = "constant",
    cval = 0.0,
    horizontal_flip = True,
    vertical_flip = True, 
)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.15) # split data

train_gen = train_datagen.flow(
    x_train,
    y_train,
    batch_size = 32,
    shuffle = True
)

####################
# Specifying model #
####################

inp = Input(shape = (150, 150, 3))

model_base = DenseNet121(
    include_top = False,
    weights = "./input/DenseNet-BC-121-32-no-top.h5",
    input_tensor = inp
)

x = GlobalAveragePooling2D()(model_base.output) # adding classifier top
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(5, activation = "softmax")(x)

model = Model(inp, x)

model.compile(
    optimizer = Adam(lr = 1e-4),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

#################
# Fitting model #
#################

class KappaEval(Callback): # keras callback for QW-kappa loss evaluation
    
    def on_train_begin(self, logs = {}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs = {}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)

        _val_kappa = cohen_kappa_score(
            y_val.argmax(axis = 1), 
            y_pred.argmax(axis = 1), 
            weights = "quadratic"
        )

        self.val_kappas.append(_val_kappa)
        print("QWK:", _val_kappa)
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved, saving model") # save if best model so far
            self.model.save("model_best.h5")        
        
        return

kappa_metrics = KappaEval()

callbacks_list = [
    ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = 3,
        verbose = 1,
        mode = "min",
    ),
    EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 6,
        restore_best_weights = False
    ),
    kappa_metrics
]

history = model.fit_generator(
    train_gen,
    validation_data = (x_val, y_val),
    steps_per_epoch = x_train.shape[0] // 32,
    epochs = 25,
    callbacks = callbacks_list,
    verbose = 2
)

##############
# Evaluation #
##############

history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
history_df[["acc", "val_acc"]].plot()

plt.plot(kappa_metrics.val_kappas)

###############
# Submissions #
###############

model.load_weights("model_best.h5") # load best (wrt kappa) model weights 

preds = model.predict(x_test)
preds = np.argmax(preds, axis = 1)

test_df["diagnosis"] = preds
test_df.to_csv("submission.csv", index = False)
