from cleo.commands.command import Command
from deep_astro_uda.client.options import full_options
from deep_astro_uda.configs.config_functions import ConfigParser
from deep_astro_uda.data_utils.download_data import Downloader
from deep_astro_uda.settings import DATASET_NAMES, DEFAULT_CONFIG_PATH
import os
import shutil 
from deep_astro_uda.model.train import train
from deep_astro_uda.model.eval import test

# TODO: Add docstrings.
class RunCommand(Command):
    """
    Run training, testing, and inference pipeline.
    """
    name = "run"

    options = full_options

    help = ""

    def handle(self):
        self.line('Running the client')

        config_path = self.option('config-path')
        dataset = self.option('dataset')
        unknowns = self.option("unknowns-supplied")
        img_path = self.option("image-path-text")
        output_dir = self.option("output-directory")
        data_type = self.option("data-type")
        domain = self.option("domain-type")

        if dataset is None or dataset not in DATASET_NAMES:
            raise ValueError("You must provide a value for the dataset. Choose one of the following: astro-nn, office_home, galaxy_zoo, deep_adversaries, data.")
        elif dataset is 'data':
            # TODO:Implement check that the user-data-path (img-path) actually exists and has the right structure.
            if not os.path.exists(img_path):
                raise ValueError("Please validate that your img-path exists.")
            else:
                source_dir_path = os.path.join(img_path, "source")
                target_dir_path = os.path.join(img_path, "target")

                if not os.path.isdir(source_dir_path) and target_dir_path:
                    os.mkdir(source_dir_path)
                    os.mkdir(target_dir_path)
                    for idx, file in enumerate(os.listdir(img_path)):
                        src_path = os.path.join(file, img_path)
                        dest_path = os.path.join(file, target_dir_path)
                        if idx % 2 == 0:
                            shutil.move(src_path, dest_path)
        else:
            pass

        self.overwrite(f"The specified dataset was of type:{dataset}")

        if data_type:
            if data_type not in ['jpg', 'png', 'numpy']:
                raise ValueError("An inappropriate data type has been supplied. The only valid options are jpg, png, numpy.")
        else:
            # TODO: Implement peeking at the first image in the dataset.
            pass

        if config_path is None:
            # TODO: Implement function and add here.
            # Create the default configuration template file based on the specified dataset.
            # INSERT CREATE_CONFIG_TEMPLATE_FUNCTION at the DEFAULT_CONFIG_PATH.
            config = ConfigParser(file_path=DEFAULT_CONFIG_PATH, filename=f"{dataset}_{domain}.yaml")
            config.create_yaml_from_template(template=dataset)
        else:
            parser = ConfigParser(config_file=config_path)

        if not output_dir:
            output_path = parser.config["output_path"]
        else:
            output_path = output_dir

        self.line(f"Running the DeepAstroUDA run command. All output will be saved here: {output_path}")

        if not unknowns:
            unknowns = parser.config["nums_unknowns"]
        
        if dataset in DATASET_NAMES:
            download = Downloader(output_dir=output_path)

            download.download_data(dataset_name=dataset)


        # Input the training run.
        dataset_test, filename, n_share, unk_class, G, C1, threshold = train()

        # Input the inference command.
        test(200, dataset_test, filename, n_share, unk_class, G, C1, threshold)
