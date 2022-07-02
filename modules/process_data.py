import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
from azureml.core import Run

AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 100
IMAGE_SIZE = 128
CHANNELS = 3
CLASSES = 3
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EPOCHS = 15


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')


def read_image_from_path(path: str) -> dict:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(path, "images", "annotations/trimaps")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.where(mask == 3, np.dtype('uint8').type(0), mask)

    return {'image': image, 'segmentation_mask': mask}


def main(dataset_path):
    full_dataset = tf.data.Dataset.list_files(dataset_path + "*.jpg", seed=SEED)
    training_set = full_dataset.take(int(0.8 * len(full_dataset)))
    TRAINSET_SIZE = len(training_set)
    training_set = training_set.map(read_image_from_path)

    validation_set = full_dataset.skip(int(0.8 * len(full_dataset)))
    VALSET_SIZE = len(validation_set)
    validation_set = validation_set.map(read_image_from_path)
    # Rechnik koj gi sodrzi trening i validaciskoto mnozestvo
    dataset = {"train": training_set, "val": validation_set}

    # -- Trening mnozestvo --#
    dataset['train'] = dataset['train'].map(load_training_image, num_parallel_calls=AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validacisko mnozestvo --#
    dataset['val'] = dataset['val'].map(load_validation_image)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    model = build_model(CLASSES)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights("pets_model.h5")
    run.log('model summary', model.summary())

    num = 5
    predictions(model, dataset =  dataset['train'])

    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            # Predviduvanje na segmentacija na 1 sluchajno izbrana slika od validacisko mnozhestvo
            predictions(model, dataset['val'])
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

    training_callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint('../pets_model_after.h5', verbose=1, save_best_only=True,
                                           save_weights_only=True)
    ]

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
        fig = plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss Results')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        run.log_image('Training and Validation Loss Results', fig)

        # Evaluacija na modelot na validacisko mnozhestvo vo 10 chekori
        res = model.evaluate(dataset["val"], batch_size=BATCH_SIZE, steps=10)
        print("test loss, test acc:", res)
        run.log_list('test loss, test acc:', res)


@tf.function
def normalize_input(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask


@tf.function
def load_training_image(data_dict: dict) -> tuple:
    image = tf.image.resize(data_dict['image'], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(data_dict['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE))

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    transformed_image, transformed_mask = normalize_input(image, mask)

    return transformed_image, transformed_mask


@tf.function
def load_validation_image(data_dict: dict) -> tuple:
    image = tf.image.resize(data_dict['image'], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(data_dict['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE))

    normalized_image, normalized_mask = normalize_input(image, mask)

    return normalized_image, normalized_mask


def visualize_sample(image_list):
    fig = plt.figure(figsize=(18, 18))

    title = ['Input Image', 'Ground truth', 'Predicted Mask']

    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image_list[i]))
        plt.axis('off')
    plt.show()
    run.log_image('prediction', plot=fig)


def build_model(output_channels,):

    input_size = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    initializer = 'uniform'

    # ------------------------------------------------------------ #
    # --------------------------MODEL----------------------------- #
    # ------------------------------------------------------------ #

    # -- ENCODER -- #
    encoder = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)


    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]

    layers = [encoder.get_layer(name).output for name in layer_names]
    contraction = tf.keras.Model(inputs=encoder.input, outputs=layers)
    contraction.trainable = False
    run.log('encoder summary', contraction.summary())

    # -- DECODER -- #

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    outs = inputs
    ups = [512, 256, 128, 64]
    # Down sempliranje preku modelot
    residuals = contraction(outs)
    outs = residuals[-1]
    residuals = reversed(residuals[:-1])

    # Up sempliranje i dodavanje skip konekcii
    for up, skip in zip(ups, residuals):
        outs = Conv2D(up, 3, activation='relu', padding='same', kernel_initializer=initializer)(
            UpSampling2D(size=(2, 2))(outs))
        concat = tf.keras.layers.Concatenate()
        outs = concat([outs, skip])

    # Posleden sloj na modelot
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128
    outs = last(outs)

    return tf.keras.Model(inputs=inputs, outputs=outs)


# ------------------------------------------------------------ #
# --------------------------MODEL----------------------------- #
# ------------------------------------------------------------ #


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    pred_mask = tf.argmax(pred_mask, axis=-1)  # go zema brojot na kanalot vo koj e najgolemata vrednost
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def predictions(model, sample_image=None, sample_mask=None, dataset=None, num_samples=1):
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
        visualize_sample([sample_image[0], sample_mask[0],
                          pred_mask[0]])


if __name__ == '__main__':
    run = Run.get_context()
    dataset_path = run.input_datasets['training_files']
    main(dataset_path)
    run.complete()
