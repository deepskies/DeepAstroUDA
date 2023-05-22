import yaml
import os
from deep_astro_uda.settings import DATASET_NAMES, ASTRO_NN_CONFIG

class ConfigParser:
    def __init__(self, file_path, filename="config.yaml", config_file=None):

        self.__check_path_exists__(file_path)

        self.path = file_path
        self.filename = filename
        
        if config_file is None:
            self.config_file = os.path.join(self.path, self.filename)
        else:
            self.config_file = config_file

        self.config = self.parse_config()

    def parse_config(self):

        with open(self.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    @property
    def config(self):
        return self.config
    
    def create_yaml_from_template(self, template=None):
        """
            Template are the yamls from the available experiments. Values provided are the datasets available.
        """
        with open(self.config_file, 'w') as yaml_file:
            yaml_file.write(yaml.dump(variables, default_flow_style=False))

    def __variables_from_template__(self, template_name):
        """
        Loads the variables from the template.
        """
        if template_name not in DATASET_NAMES:
            raise ValueError(f"Template name must be one of {DATASET_NAMES}")
        else:
            if template_name == "astro-nn":
                return ASTRO_NN_CONFIG
            elif template_name == "office_home":
                return OFFICE_HOME_CONFIG
            elif template_name == "galaxy_zoo":
                return GALAXY_ZOO_CONFIG
            elif template_name == "deep_adversaries":
                return DEEP_ADVERSARIES_CONFIG
            else:
                return DEFAULT_CONFIG


    def __check_path_exists__(self):
        """
        Checks if the path exists. If the path doesn't exist, then create the path in the current directory.
        """

        if not os.path.exists(self.path):
            os.makedirs(self.path)
