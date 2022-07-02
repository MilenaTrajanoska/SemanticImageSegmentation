"""

    SEMINARSKA RABOTA PO PREDMETOT DIGITALNO PROCESIRANJE NA SLIKA
    TEMA: ALGORITMI ZA SEGMENTACIJA NA SLIKI BAZIRANI NA DLABOKO UCHENJE

    PODTEMA: SEMANTICKA SEGMENTACIJA NA SLIKI SO U-NET ARHITEKTURA NA NEVRONSKA MREZA

    MENTOR: VON. PROF. D-R IVICA DIMITROVSKI
    IZRABOTENA OD: MILENA TRAJANOSKA 181004

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output

AUTOTUNE = tf.data.experimental.AUTOTUNE

#za generiranje na istite sluchajni broevi sekoj pat
SEED = 100

#relativna pateka do direktoriumot so originalni sliki
dataset_path = "C:/Users/mimi_/PycharmProjects/SemanticImageSegmentation/Oxford_pets/images/"

# Goleminata na slikite koja kje se koristi
# Slikite imat golemaina od 128x128 i 3 kanali za RGB
# Maskite imaat eden kanal

IMAGE_SIZE = 128
CHANNELS = 3

# Podatochnoto mnozhestvo Oxford-IIIT Pet Dataset ima 3 klasi:
# 1. Pikseli koi pripagjaat na milenicheto
# 2. Pikseli od konturata na milenicheto
# 3. Pozadina
# potrebna e transformacija na vrednostite na klasite vo 0, 1, 2

CLASSES = 3

def read_image_from_path(path: str) -> dict:
    """ Vcituvanje na slika i nejzina maska
    Rezultatot koj se vrakja e recnik od originalnata slika i maskata

    Parametri
    ----------
    path : str
        Pateka na originalnata slika

    Povraten tip
    -------
    dict
        Recnik koj gi sodrzi originalnata slika i segmentiranata ground truth slika
    """
    # Vchituvanje na slikata i dekodiranje vo soodveten format
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    # Zamena na direktoriumite na originalnata slika so lokacijata na segmentiranata slika i zamena na formatot
    mask_path = tf.strings.regex_replace(path, "images", "annotations/trimaps")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    
    # Vchituvanje na segmentiranata slika i dekodiranje vo soodvetniot format
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    # Vo podatochnoto mnozhestvo, trite klasi se oznacheni so vrednosti 1, 2 i 3
    # Vo keras se pochnuva so oznachuvanje na klasite od 0, pa bidejkji brojot na klasi e 3
    # Keras gi ochekuva vrednostite 0, 1 i 2, zatoa klasata oznachena kako 3 samo ja menuvame so klasa 0
    mask = tf.where(mask == 3, np.dtype('uint8').type(0), mask)

    # Rezultat
    return {'image': image, 'segmentation_mask': mask}

# Go vchituvame celosnoto podatochno mnozhehstvo
# Prvite 80% ke gi koristime za treniranje na modelot 
# Ostanatite 20% ke gi koristime za validacija na modelot

full_dataset = tf.data.Dataset.list_files(dataset_path  + "*.jpg", seed=SEED)
training_set = full_dataset.take(int(0.8*len(full_dataset)))
TRAINSET_SIZE = len(training_set)
training_set = training_set.map(read_image_from_path)

validation_set = full_dataset.skip(int(0.8*len(full_dataset)))
VALSET_SIZE = len(validation_set)
validation_set =validation_set.map(read_image_from_path)

@tf.function
def normalize_input(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    """ Normaliziranje na vrednostite na pikselite od vleznata slika vo interval [0.0, 1.0]
    
    Tehnika na regularizacija na vlezot vo nevronskata mrezha so cel pobrza konvergencija

    Parametri
    ----------
    image : tf.Tensor
        Tensorflow tenzor koj sodrzi slika so golemina [IMAGE_SIZE,IMAGE_SIZE,3]
    mask : tf.Tensor
        Tensorflow tenzor koj sodrzi segmentirana ground truth slika so golemina [IMAGE_SIZE,IMAGE_SIZE,1]
        
    Rezultat
    -------
    tuple
        Normalizirana slika i nepromeneta maska
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask

@tf.function
def load_training_image(data_dict: dict) -> tuple:
    """ Transformacii vrz trening slikite i nivnite soodvetni segmentirani sliki

    Parametri
    ----------
    data_dict : dict
        Recnik koj gi sodrzi originalnata slika i segmentiranata slika

    Rezultat
    -------
    tuple
        Modificirana originalna slika i segmentirana slika
    """
    image = tf.image.resize(data_dict['image'], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(data_dict['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE))
    # data augmentation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    transformed_image, transformed_mask = normalize_input(image, mask)

    return transformed_image, transformed_mask

@tf.function
def load_validation_image(data_dict: dict) -> tuple:
    """Normaliziranje i menuvanje na goleminata na validaciskite sliki i segmentacii

    Parametri
    ----------
    data_dict : dict
        Rechnik od originalnata slika i segmentiranata slika

    Rezultat
    -------
    tuple
        Modificirana originalna slika i segmentirana slika
    """
    image = tf.image.resize(data_dict['image'], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(data_dict['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE))

    normalized_image, normalized_mask = normalize_input(image, mask)

    return normalized_image, normalized_mask

# Golemina na serija
BATCH_SIZE = 32

# Golemina na bufferot za sluchajno menuvanje na rasporedot
# Celta e invarijantnost na mrezata na redosledot na vlezot
BUFFER_SIZE = 1000

# Rechnik koj gi sodrzi trening i validaciskoto mnozestvo
dataset = {"train": training_set, "val": validation_set}

# -- Trening mnozestvo --#
dataset['train'] = dataset['train'].map(load_training_image, num_parallel_calls=AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Validacisko mnozestvo --#
dataset['val'] = dataset['val'].map(load_validation_image)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

def visualize_sample(image_list):
    """Prikaz na originalnata slika, ground truth segmentacijata i predvidenata segmentacija

    Parametri
    ---------
        image_list : [type]
            Lista od slikite koi treba da se prikazhat
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'Ground truth', 'Predicted Mask']

    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image_list[i]))
        plt.axis('off')
    plt.show()


# Dimenzii na vlezot
input_size = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

initializer = 'uniform'

# ------------------------------------------------------------ #
# --------------------------MODEL----------------------------- #
# ------------------------------------------------------------ #

# Koristime transfer learning na vekje treniran model MovlieNetV2
# Ke go koristime samo encoderot od ovoj model i tie sloevi ke gi zamrzneme so postavuvnje na trainable = False

# ------------------------------------------------------------ #
# --------------------------MODEL----------------------------- #
# ------------------------------------------------------------ #

# -- ENCODER -- #
encoder = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Sloevi koi ke gi koristime od enkoderot
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

layers = [encoder.get_layer(name).output for name in layer_names]

# Kreirame model za ekstrakcija na karakteristiki, odbranite sloevi g stavame kako izlezi od modelot
contraction = tf.keras.Model(inputs=encoder.input, outputs=layers)

# Gi zamrznuvame sloevite da ne se azuriraat parametrite pri treningot
contraction.trainable = False
# Pechatime rezimena modelot
contraction.summary()

# -- DECODER -- #

# Broj na kerneli za konvolucija vo dekoderot
ups = [512, 256, 128, 64]

# Funkcija koja ja kreira modificiranata u-net mrezha
def create_unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  outs = inputs

  # Down sempliranje preku modelot
  residuals = contraction(outs)
  outs = residuals[-1]
  residuals = reversed(residuals[:-1])

  # Up sempliranje i dodavanje skip konekcii
  for up, skip in zip(ups, residuals):
    outs = Conv2D(up, 3, activation='relu', padding='same', kernel_initializer=initializer)(UpSampling2D(size=(2, 2))(outs))
    concat = tf.keras.layers.Concatenate()
    outs = concat([outs, skip])

  # Posleden sloj na modelot
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128
  outs = last(outs)

  return tf.keras.Model(inputs=inputs, outputs=outs)

# ------------------------------------------------------------ #
# --------------------------MODEL----------------------------- #
# ------------------------------------------------------------ #

# Go kreirame modelot
model = create_unet_model(CLASSES)

# Go kompajlirame modelot so optimizer Adam so rata na uchenje 0.0001 i zaguba Sparse Categorical Crossentropy
model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Gi vchituvame tezhinite zachuvani od vekje istreniraniot model
model.load_weights("pets_model.h5")
# Pechatime rezime za modelot
model.summary()

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """ Ja vrakjame maskata za predviduvanjeto

    Parametri
    ----------
    pred_mask : tf.Tensor [IMAGE_SIZE, IMAGE_SIZE, CLASS] tenzor so tri kanali

    Rezultat
    -------
    tf.Tensor
          [IMAGE_SIZE, IMAGE_SIZE, 1] maska so najdobroto predviduvanje so eden kanal
    """
    # pred_mask -> [IMAGE_SIZE, SIZE, N_CLASS]
    # imame po edno predviduvanje za sekoja mozhna klasa
    # za da go zememe najdobroto predviduvanje pravime argmax
    pred_mask = tf.argmax(pred_mask, axis=-1) #go zema brojot na kanalot vo koj e najgolemata vrednost
    # pred_mask stanuva [IMAGE_SIZE, IMAGE_SIZE]
    # no matplotlib bara format [IMAGE_SIZE, IMAGE_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def predictions(dataset=None, num_samples=1):
    """Prikazhuvanje na primer predviduvanje

    Parametri
    ----------
    dataset : [type], opcionalen

    num_samples : int, opcionalen, default 1

    """
    if dataset:
        for image, mask in dataset.take(num_samples):
            prediction = model.predict(image)
            visualize_sample([image[0], mask[0], create_mask(prediction)[0]])
    else:
        # [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]

        img = sample_image[0][tf.newaxis, ...]
        # img -> [1, IMAGE_SIZE, IMAGE_SIZE, 3]
        pred = model.predict(img)
        # pred -> [1, IMAGE_SIZE, IMAGE_SIZE, N_CLASS]
        pred_mask = create_mask(pred)
        # pred_mask -> [1, IMAGE_SIZE, IMAGE_SIZE, 1]
        #vizuelizacija na primerokot
        visualize_sample([sample_image[0], sample_mask[0],
                        pred_mask[0]])

# Broj na primer predviduvanja
print("Enter number of sample predictions to visualize:")
num = int(input())
# Zemame edna primer slika za vizuelizacija vo callback
for image, mask in dataset['train'].take(num):
    sample_image, sample_mask = image, mask
    predictions()

# Definirame broj na epohi
EPOCHS = 15

# Definirame broj na cekori po epoha za trening i validacija
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

# Definirame svoj callback za prikazhuvanje na eden primerok posle sekoja epoha od treniranjeto
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # Predviduvanje na segmentacija na 1 sluchajno izbrana slika od validacisko mnozhestvo
        predictions(dataset['val'])
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

training_callbacks = [
    # prikazhuvanje na primerok
    DisplayCallback(),
    # early stopping ako nemame podobruvanje na tochnosta na validacisko mnozhestvo 3 pati
    tf.keras.callbacks.EarlyStopping(patience=3, monitor="val_accuracy"),
    # gi zachuvuvame nauchenite tezhini po sekoja epoha vo koja imame podobruvanje
    tf.keras.callbacks.ModelCheckpoint('../pets_model_after.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

print("Training or validation? (t/v)")
#train = str.lower(input())
train = 'v'

if train == 't':
    # Treniranje na modelot na cpu
    with tf.device("/cpu:0"):
        model_history = model.fit(dataset['train'],
                                  epochs=EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VALIDATION_STEPS,
                                  validation_data=dataset['val'],
                                  callbacks=training_callbacks)

        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        epochs = range(EPOCHS)
        # Grafichki prikaz na zaguba nad trening i validacisko mnozhestvo
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss Results')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

elif train == 'v':
    # Evaluacija na modelot na validacisko mnozhestvo vo 10 chekori
    res = model.evaluate(dataset["val"], batch_size=BATCH_SIZE, steps=10)
    print("test loss, test acc:", res)