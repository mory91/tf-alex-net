import model as alexnet
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MAX_BUFFER_SIZE = 5000

strategy = tf.distribute.MultiWorkerMirroredStrategy()
def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227, 227))
    return image, label

def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    validation_images, validation_labels = train_images[:5000], test_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    return train_ds, test_ds, validation_ds

def train():
    train_ds, test_ds, validation_ds = load_dataset()
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    train_ds = (
        train_ds.map(process_images)
                .shuffle(buffer_size=min(MAX_BUFFER_SIZE, train_ds_size))
                .batch(batch_size=32, drop_remainder=True)
    )
    test_ds = (
        test_ds.map(process_images)
               .shuffle(buffer_size=min(MAX_BUFFER_SIZE, test_ds_size))
               .batch(batch_size=32, drop_remainder=True)
    )
    validation_ds = (
        validation_ds.map(process_images)
                     .shuffle(buffer_size=min(MAX_BUFFER_SIZE, validation_ds_size))
                     .batch(batch_size=32, drop_remainder=True)
    )

    with strategy.scope():
        model = alexnet.create_model()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        model.summary()

    model.fit(
        train_ds,
        epochs=50,
        validation_data=validation_ds,
        validation_freq=2
    )

    model.evaluate(test_ds)

if __name__ == "__main__":
    train()
