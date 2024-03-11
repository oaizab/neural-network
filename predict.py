from keras import saving, datasets
import numpy as np
import matplotlib.pyplot as plt

# load training and testing data
data = datasets.fashion_mnist.load_data()
(test_images, test_labels) = data[1]

# normalize data
test_images = test_images / 255.0

model = saving.load_model("model.keras")

# change this to select an image for prediction
index = 20

image = test_images[index]
label = test_labels[index]
prediction = model.predict(np.expand_dims(image, 0))[0]

# these are the classifier's label names for the output layer
names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.yticks([])
plt.xticks([])

plt.subplot(1, 2, 2)
plt.yticks(range(10), names)
bar_plot = plt.barh(range(10), prediction, color="red")
bar_plot[label].set_color("green")

plt.subplots_adjust(bottom=0.2)
plt.show()
