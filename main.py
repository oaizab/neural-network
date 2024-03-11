from keras import Sequential, layers, losses, optimizers, metrics, datasets

# load training and testing data
data = datasets.fashion_mnist.load_data()
train, test = data
(train_images, train_labels) = train
(test_images, test_labels) = test

# normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# configure the model
model = Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()],
)

# train the model
model.fit(train_images, train_labels, epochs=10)

model.save("model.keras")

# test the model
model.evaluate(test_images, test_labels, verbose=2)
