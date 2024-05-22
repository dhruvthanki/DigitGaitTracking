from PIL import Image

def overlay_images(image_paths, transparencies):
    """
    Overlay a series of images with specified transparencies.

    :param image_paths: List of paths to the images.
    :param transparencies: List of transparency values (0 to 255) for each image.
    :return: The resulting overlaid image.
    """
    base_image = Image.open(image_paths[0]).convert("RGBA")
    base_image = base_image.copy()  # Ensure we don't modify the original image

    for i, image_path in enumerate(image_paths[1:]):
        overlay_image = Image.open(image_path).convert("RGBA")
        # Adjust transparency
        alpha = transparencies[i + 1]
        overlay_image.putalpha(alpha)
        # Composite the images
        base_image = Image.alpha_composite(base_image, overlay_image)

    return base_image

# Example usage
image_paths = ["/home/dhruv/Pictures/output_image2gait.png",
               "/home/dhruv/Pictures/output_image3gait.png",
               "/home/dhruv/Pictures/output_image4gait.png",
               "/home/dhruv/Pictures/output_image5gait.png",
               "/home/dhruv/Pictures/output_image6gait.png",
               "/home/dhruv/Pictures/output_image7gait.png",
               "/home/dhruv/Pictures/output_image8gait.png",]

start = 225
end = 128
num_values = len(image_paths)-1
step = (end - start) / (num_values - 1)
transparencies = [int(start + i * step) for i in range(num_values)]
transparencies.append(64)  # Full opacity for the last image
# transparencies = [255, 128, 64]  # Full opacity for the first image, half for the second, and quarter for the third

result_image = overlay_images(image_paths, transparencies)
result_image.show()  # Display the resulting image
result_image.save("result.png")  # Save the resulting image
