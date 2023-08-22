import numpy as np
from PIL import Image
from t_ngd_cifar10 import test_classifier, linearize_pixels
from t_ngd_cifar10 import create_f
from ngd_attacks import num_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch


def generate_random_noise(shape, intensity=0.05):
    return np.random.uniform(-intensity, intensity, shape)


def generate_random_image(base_image, alpha_shift, noise_intensity=0.05):
    alpha = boundary_alpha + alpha_shift
    interpolated_image = (1 - alpha) * frog_sample + alpha * ship_array
    random_noise = generate_random_noise(interpolated_image.shape, noise_intensity)
    random_image = interpolated_image + random_noise
    return np.clip(random_image, 0, 255)




# def pppgd_improved(f, x, num_steps=10, initial_step_size=0.5, momentum=0.9, target_confidence=0.5):
#     conf = f(x)
#     print("Initial confidence is {}".format(conf))
#
#     if conf >= target_confidence:
#         print("Image already has confidence >= target confidence")
#         return x
#
#
#     step_size = initial_step_size
#     grad = num_grad(f, x)  # Ensure that num_grad function returns the gradient with respect to the confidence score
#     sign_data_grad = torch.sign(torch.from_numpy(grad))
#     update = torch.zeros_like(sign_data_grad)
#
#     for i in range(num_steps):
#         x = torch.from_numpy(x)
#         update = momentum * update + step_size * sign_data_grad
#         # print(x.shape)
#         # print(update.shape)
#         x = x + update
#         x = x.detach().numpy()
#         conf = f(x)
#         print("Step {}, confidence {}".format(i + 1, conf))
#
#         if conf >= target_confidence:
#             print("Reached target confidence!")
#             break
#
#         step_size *= 0.99  # learning rate decay
#
#     conf = f(x)
#     print("Final confidence is {}".format(conf))
#
#     return x

def num_ascent(f, x, threshold=0.5, max_iterations=100, step_size=0.8):
    conf = f(x)
    print("Initial confidence is {}".format(conf))

    iteration = 0
    while conf < threshold and iteration < max_iterations:
        grad = num_grad(f, x)
        x += step_size * grad
        conf = f(x)

        print("Iteration {}: Confidence {}".format(iteration + 1, conf))
        iteration += 1

    print("Final confidence is {}".format(conf))
    return x


# Note: Ensure that the 'f' function returns the confidence score of the image being in the desired class.
# The 'num_grad' function should compute the gradient of this confidence score with respect to the image.

def optimize_confidence_to_target(image, target_class, threshold=0.5, max_iterations=100, step_size=0.8):
    """
    Optimize the confidence of an image to be closer to a target class.

    Args:
    - image: The image to optimize.
    - target_class: The class we want to move the image's confidence towards.
    - num_steps, initial_step_size, momentum, target_confidence: Parameters for the PGD method.

    Returns:
    - Optimized image.
    """
    # h, w, _ = image.shape
    # f_target = create_f(h, w, target_class)
    #
    # optimized_image = pppgd_improved(f_target, image, num_steps=num_steps,
    #                                  initial_step_size=initial_step_size, momentum=momentum,
    #                                  target_confidence=target_confidence)
    # return optimized_image
    h, w, _ = image.shape
    f_target = create_f(h, w, target_class)  # Here, we assume net is globally defined. Modify as needed.

    optimized_image = num_ascent(f_target, image, threshold=threshold,
                                 max_iterations=max_iterations, step_size=step_size)
    return optimized_image


# Now, you can call this function for each side of the decision boundary.
# For example:
# optimized_image_1 = optimize_confidence_to_target(random_image_1, target_class=class_of_representative_sample_1)
# optimized_image_2 = optimize_confidence_to_target(random_image_2, target_class=class_of_representative_sample_2)

# Choosing representative samples for two classes
frog_sample = Image.open('./data/cifar_pictures/NO.1class6frog.jpg')
ship_sample = Image.open('./data/cifar_pictures/NO.2class8ship.jpg')


# Display and classify the representative samples
def display_and_classify(sample, label):
    h, w, img_array = linearize_pixels(sample)
    identified_class,class_index,confidence = test_classifier(h, w, img_array,return_class_index = True, return_confidence = True)
    plt.imshow(sample)
    plt.title(f"{label} Representative Sample, Classified as: {identified_class}")
    plt.show()
    return identified_class, class_index # Return the identified class for further comparison


frog_class,frog_class_index = display_and_classify(frog_sample, "frog")
ship_class, ship_class_index = display_and_classify(ship_sample, "ship")

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
    random_image_side_1 = generate_random_image(frog_sample, -0.05)
    random_image_side_2 = generate_random_image(ship_sample, 0.05)

    # Classify and display these random images
    class_1 = display_and_classify(Image.fromarray(np.uint8(random_image_side_1)), "Random Image Side 1")
    class_2 = display_and_classify(Image.fromarray(np.uint8(random_image_side_2)), "Random Image Side 2")

    print(f"Random image from side 1 is classified as: {class_1}")
    print(f"Random image from side 2 is classified as: {class_2}")
    optimized_image_1 = optimize_confidence_to_target(random_image_side_1, target_class=frog_class_index)
    h, w, img_array_1 = linearize_pixels(Image.fromarray(np.uint8(optimized_image_1)))
    class_1_optimized, conf_1_optimized = test_classifier(h, w, img_array_1, return_confidence=True)
    print(f"Optimized image 1 is classified as: {class_1_optimized} with confidence {conf_1_optimized}")

    optimized_image_2 = optimize_confidence_to_target(random_image_side_2, target_class=ship_class_index)
    h, w, img_array_2 = linearize_pixels(Image.fromarray(np.uint8(optimized_image_2)))
    class_2_optimized, conf_2_optimized = test_classifier(h, w, img_array_2, return_confidence=True)
    print(f"Optimized image 2 is classified as: {class_2_optimized} with confidence {conf_2_optimized}")
else:
    print("No decision boundary found.")


