import numpy as np
import os
from glob import glob
from skimage.measure import label, regionprops
import tifffile as tiff


def count_multiplicity(image_path: str, threshold_value: int):
    """
    Count multiplicity in a single image using tifffile.

    Parameters:
    image_path (str): Path to the TIFF image.
    threshold_value (int): Threshold value to consider a pixel as part of a cluster.

    Returns:
    list[int]: List of multiplicities (cluster sizes).
    """
    image = tiff.imread(image_path)

    # Ensure it's a 2D grayscale image
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

    # Apply threshold to detect clusters
    binary_image = image >= threshold_value

    # Label connected regions
    labeled_image = label(binary_image, connectivity=2)

    # Calculate multiplicity for each cluster
    multiplicity_counts = [region.area for region in regionprops(labeled_image)]

    return multiplicity_counts


def process_images_in_folder(folder_path: str, threshold_value: int = 1):
    """
    Process all TIFF images in a folder to count multiplicities.

    Parameters:
    folder_path (str): Path to the folder containing TIFF images.
    threshold_value (int, optional): Threshold value to consider a pixel as part of a cluster (default=1).

    Returns:
    list[list[int]]: List of multiplicity counts for each image.
    """
    image_paths = glob(os.path.join(folder_path, "*.tiff"))

    if not image_paths:
        print(f"No TIFF files found in folder: {folder_path}")
        return []

    all_multiplicities = [count_multiplicity(image_path, threshold_value) for image_path in image_paths]

    return all_multiplicities


def average_multiplicity_with_error(all_multiplicities):
    """
    Calculate the average multiplicity and standard error across all images.

    Parameters:
    all_multiplicities (list of list): List of multiplicity counts for each image.

    Returns:
    avg_multiplicity (float): The average multiplicity across all images.
    sem (float): The standard error of the mean for the multiplicity.
    """
    # Flatten the list of multiplicities
    flattened_multiplicities = [multiplicity for sublist in all_multiplicities for multiplicity in sublist]

    if flattened_multiplicities:
        avg_multiplicity = np.mean(flattened_multiplicities)
        # Standard deviation of the multiplicities
        std_dev = np.std(flattened_multiplicities, ddof=1)  # ddof=1 for sample std deviation
        # Number of multiplicities
        n = len(flattened_multiplicities)
        # Standard error of the mean (SEM)
        sem = std_dev / np.sqrt(n)
    else:
        avg_multiplicity = 0
        sem = 0  # Avoid division by zero if no clusters are found

    return avg_multiplicity, sem


if __name__ == "__main__":
    # Ask user for the folder path and threshold value
    folder_path = input("Enter the folder path containing TIFF images: ")

    threshold_value = 1

    # Process images
    all_multiplicities = process_images_in_folder(folder_path, threshold_value)

    if all_multiplicities:
        avg_multiplicity, sem = average_multiplicity_with_error(all_multiplicities)

        for i, multiplicities in enumerate(all_multiplicities):
            print(f"Image {i + 1}:")
            print(f"  Multiplicities: {multiplicities}")
            print(f"  Number of clusters: {len(multiplicities)}")
            print(f"  Average Multiplicity: {np.mean(multiplicities) if multiplicities else 0}")

        print(f"Overall Average Multiplicity across all images: {round(avg_multiplicity, 2)}")
        print(f"Standard Error of the Mean (SEM): {sem}")
    else:
        print("No images processed.")
