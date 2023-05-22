from astroNN.datasets import load_galaxy10
import os
from PIL import Image
import numpy as np

class Downloader:

    def __init__(self, output_dir):
        
        self.output_dir = output_dir

    def download_data(self, dataset_name):
        """
        Downloads the specified dataset from its source. If dataset includes more than one piece, a list is returned.
        """
        if dataset_name == "astro-nn":
            self.__download_astro_nn__()
        elif dataset_name == "office_home":
            self.__download_office_home__()
        elif dataset_name == "galaxy_zoo":
            self.__download_galaxy_zoo__()
        elif dataset_name == "deep_adversaries":
            self.__download_deep_adversaries__()
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported. Only the following are available: astro-nn, office_home, galaxy_zoo, deep_adversaries.")

    def __download_astro_nn__(self):
        """Comment Container. Sample - whether or not this download is taking place for the demo"""

        # Extract the images and labels.
        images,labels = load_galaxy10()

        # Create the original and simulated folders in the output directory.
        original_dir = os.path.join(self.output_dir, "original")
        simulated_dir = os.path.join(self.output_dir, "simulated")

        os.mkdir(original_dir)

        os.mkdir(simulated_dir)

        # Create the label folders in the output directory.
        self.__create_folders_from_labels__(labels, original_dir)
        self.__create_folders_from_labels__(labels, simulated_dir)

        # Save the images in their respective folders. Also, apply noise to the simulated images.
        for img, label, idx in zip(images, labels, range(len(images))):

            img_array = img.astype('uint8')
            noise = np.random.poisson(img_array * 0.5)

            # Apply the noise to the image array
            noisy_image_array = img_array + noise

            # Clip the pixel values to the valid range
            noisy_image_array = np.clip(noisy_image_array, 0, 255)

            img_out = Image.fromarray(np.uint8(noisy_image_array), 'RGB')
            if idx % 2 == 0:
                dir = original_dir
            else:
                dir = simulated_dir

            save_path = os.path.join(dir, str(label), str(idx)+'.jpg')
            img_out.save(save_path, "JPEG")
            
    def __download_office_home__(self):
        pass

    def __download_galaxy_zoo__(self):
        pass

    def __download_deep_adversaries__(self):
        pass
    
    def __create_folders_from_labels__(self, labels, directory):
        """
        Creates folders in the output directory based on the labels provided.
        """
        for label in labels:
            path = os.path.join(directory, str(label))
            if not os.path.exists(path):
                os.mkdir(path)

