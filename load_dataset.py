import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset"

# Load raw training dataset
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Load raw validation dataset
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Extract class names before mapping
class_names = train_ds_raw.class_names

# Normalization layer to scale pixels to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Apply normalization and performance optimizations
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y)) \
                       .cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y)) \
                   .cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Visualize a batch of 9 images
def show_sample(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            label_index = tf.argmax(labels[i]).numpy()
            plt.title(class_names[label_index])
            plt.axis("off")
    plt.show()

# Show sample from the dataset
show_sample(train_ds, class_names)
