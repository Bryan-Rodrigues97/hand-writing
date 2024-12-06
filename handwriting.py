import tensorflow as tf
import idx2numpy

# Arquivos emnist
TRAIN_IMAGES_PATH = 'emnist/emnist-byclass-train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'emnist/emnist-byclass-train-labels-idx1-ubyte'
TEST_IMAGES_PATH = 'emnist/emnist-byclass-test-images-idx3-ubyte'
TEST_LABELS_PATH = 'emnist/emnist-byclass-test-labels-idx1-ubyte'

NUM_CLASSES = 62

# Carregamento  do dataset
def load_emnist_images(image_file, label_file):
    images = idx2numpy.convert_from_file(image_file)
    labels = idx2numpy.convert_from_file(label_file)
    images = images / 255.0
    images = images.reshape(images.shape[0], 28, 28, 1)
    return images, labels

# Pré processamento
def preprocess(images, labels):                           
    images = tf.image.rot90(images, k=3)                # Rotação de 270 graus
    images = tf.image.flip_left_right(images)           # Flip horizontal
    labels = tf.one_hot(labels, depth=NUM_CLASSES)      # Converter rótulos para one-hot encoding
    return images, labels

# Criar uma rede neural convolucional
def createModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")
    ])

    return model

if __name__ == "__main__":
    # Dados de treino e teste
    x_train, y_train = load_emnist_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
    x_test, y_test = load_emnist_images(TEST_IMAGES_PATH, TEST_LABELS_PATH)
    
    # Criar pipeline de dados para treinamento
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(preprocess).batch(64).shuffle(10000).prefetch(tf.data.AUTOTUNE)
    
    # Criar pipeline de dados para teste (sem augmentação)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=62))).batch(64).prefetch(tf.data.AUTOTUNE)

    model = createModel()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_dataset, epochs=10)
    model.evaluate(test_dataset, verbose=2)

    # Salvar o modelo em um arquivo
    model.save("emnist.h5")  