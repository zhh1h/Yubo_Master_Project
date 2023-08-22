import numpy as np
from PIL import Image
from t_ngd_cifar10 import test_classifier, linearize_pixels
import matplotlib.pyplot as plt


def generate_random_noise(shape, intensity=0.05):
    return np.random.uniform(-intensity, intensity, shape)


def generate_random_image(base_image, alpha_shift, noise_intensity=0.05):
    alpha = boundary_alpha + alpha_shift
    interpolated_image = (1 - alpha) * frog_sample + alpha * ship_array
    random_noise = generate_random_noise(interpolated_image.shape, noise_intensity)
    random_image = interpolated_image + random_noise
    return np.clip(random_image, 0, 255)


# Choosing representative samples for two classes
frog_sample = Image.open('./data/cifar_pictures/NO.1class6frog.jpg')
ship_sample = Image.open('./data/cifar_pictures/NO.2class8ship.jpg')


# Display and classify the representative samples
def display_and_classify(sample, label):
    h, w, img_array = linearize_pixels(sample)
    identified_class = test_classifier(h, w, img_array)
    plt.imshow(sample)
    plt.title(f"{label} Representative Sample, Classified as: {identified_class}")
    plt.show()
    return identified_class  # Return the identified class for further comparison


display_and_classify(frog_sample, "frog")
display_and_classify(ship_sample, "ship")

# Convert these samples to numpy arrays
frog_sample = np.array(frog_sample)
ship_array = np.array(ship_sample)

# Perform linear interpolation between these two samples
steps = 25
alpha_values = np.linspace(0, 1, steps)
interpolated_samples = [(1 - alpha) * frog_sample + alpha * ship_array for alpha in alpha_values]

boundary_alpha = None
prev_class = None

# Feed the interpolated samples into the model and observe the output
for idx, sample in enumerate(interpolated_samples):
    h, w, img_array = linearize_pixels(Image.fromarray(np.uint8(sample)))
    identified_class = test_classifier(h, w, img_array)

    # Display interpolated sample and model's output
    plt.imshow(np.uint8(sample))
    plt.title(f"Alpha: {alpha_values[idx]}, Classified as: {identified_class}")
    plt.show()

    if prev_class is not None and identified_class != prev_class:
        boundary_alpha = alpha_values[idx]
        break
    prev_class = identified_class

# Check if boundary_alpha was found
if boundary_alpha is not None:
    print(f"Found decision boundary at alpha: {boundary_alpha}")

    # Generate random images on either side of the decision boundary
    random_image_side_1 = generate_random_image(frog_sample, -0.03)
    random_image_side_2 = generate_random_image(ship_sample, 0.03)

    # Classify and display these random images
    class_1 = display_and_classify(Image.fromarray(np.uint8(random_image_side_1)), "Random Image Side 1")
    class_2 = display_and_classify(Image.fromarray(np.uint8(random_image_side_2)), "Random Image Side 2")

    print(f"Random image from side 1 is classified as: {class_1}")
    print(f"Random image from side 2 is classified as: {class_2}")
else:
    print("No decision boundary found.")