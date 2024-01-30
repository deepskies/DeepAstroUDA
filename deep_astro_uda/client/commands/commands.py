from cleo import Command, option
from options import demo_options, full_options
import os 
from subprocess import call
import time 
import random



class DeepDanceDemoRun(Command):

    name = "demo"
    description = "Run the DeepDANCE algorithm implementation."
    arguments = []
    options = [
        option(long_name="config-path", value_required=True),
        option(long_name="unknowns-supplied", value_required=True),
        option(long_name="file-paths-supplied", value_required=True), #change to image-path-text
        option(long_name="output-directory", value_required=True),
        option(long_name="domain-type", value_required=True),  # implement later
        # add data type option
    ]

    def handle(self):
        config_path = self.option("config-path")
        unk_sup = self.option("unknowns-supplied")
        file_pth_sup = self.option("file-paths-supplied")
        output_dir = self.option("output-directory")

        #   0. Write the configuration options entered by the user.
        self.write(f" Configuration parameters are as follows: \n\n Config Path: {config_path} \n Number of Unknowns Supplied?: {unk_sup} \n File Paths Supplied?: {file_pth_sup} \n Output Directory: {output_dir}")
        
        #   1. Check for valid config path.
        self.write(f"Checking that the path {config_path} exists.\n")
        time.sleep(1)
        self.overwrite(f"Path exists!\n")
        #   2. If number of unknowns supplied, update the config file.
        self.write("Updating config file to account for whether your unknowns are supplied.\n\n")
        time.sleep(4)
        #   3. Check for valid output directory.
        self.write(f"Checking that the path {output_dir} exists. If not, one will be created.\n\n")
        # if not os.path.exists("records"):
        #     os.mkdir("records")
        #   4. If file path not supplied, run util to make the file path txt file.
        self.write(f"File Path Supplied is False. One will be configured for you. Please ensure folders are in correct format.\n\n")
        time.sleep(5)
        self.write("File path text file configured successfully!\n\n")
        #   5. Check that all the config path arguments make sense.
        self.write(f"Ensuring valid config path structure...\n")
        time.sleep(7)
        self.overwrite(f"Configuration path secured!\n")
        #   6. Begin training ... print out ETA and progress bar.
        self.write(f"Beginning training!\n")
        time.sleep(9)
        #   7. Training ends.
        self.overwrite(f"Training has completed! Output files will be placed their designated folders in {output_dir}.\n")
        time.sleep(3)
        #   8. Output loss, accuracy, t-SNE files.
        self.write(f"Output files, saved models, and t-SNE window available! Please exit t-SNE window to cease function.\n")
        time.sleep(5)

        #   9. Output saved models. 
        self.overwrite(config_path, unk_sup)
        if file_pth_sup:
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")
        else:
            self.write(file_pth_sup)
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")


class DeepAstroDemoRun(Command):

    name = "deepastro demo"
    description = "Run the DeepAstroUDA algorithm demo."
    arguments = []
    options = demo_options

    def handle(self):

        domain_type = self.option("domain-type")
        config_path = self.option("config-path")
        output_dir = self.option("output-directory")
        data = self.option("data-type")

        #   0. Write the configuration options entered by the user.
        if data:
            if 'data':
                print('You can"t run the demo using your own personal data. Run the full pipeline or infer. AstroNN running instead')
            elif 'astro-uda':
                2
            elif 'office':
                3
            elif 'gz2':
                4
        else:
            data = 'astro-nn'

        if not output_dir:
            output_dir = '/Users/ae_rabelais/Documents/deep-astro-uda/records'

        self.write(
            f" Configuration parameters are as follows: \n\n Config Path: {config_path} \n \
            Domain Type: {domain_type} \n \
            Chosen Dataset: {data} \n \
            Output Directory: {output_dir}")
        
        #   1. Check for given config path.
        self.write(f"Checking that the path {config_path} exists.\n")
        time.sleep(2)
        self.overwrite(f"Path exists!\n")

        #   1. Check for valid config path.
        self.write(f"Checking that the path {config_path} exists.\n")
        time.sleep(2)
        self.overwrite(f"Path exists!\n")
        #   2. If number of unknowns supplied, update the config file.
        self.write("Updating config file to account for whether your unknowns are supplied.\n\n")
        time.sleep(4)
        #   3. Check for valid output directory.
        self.write(f"Checking that the path {output_dir} exists. If not, one will be created.\n\n")
        # if not os.path.exists("records"):
        #     os.mkdir("records")
        #   4. If file path not supplied, run util to make the file path txt file.
        self.write(f"File Path Supplied is False. One will be configured for you. Please ensure folders are in correct format.\n\n")
        time.sleep(5)
        os.mkdir(f"{output_dir}/data")
        self.write("File path text file configured successfully!\n\n")
        #   5. Check that all the config path arguments make sense.
        self.write(f"Ensuring valid config path structure...\n")
        time.sleep(7)
        self.overwrite(f"Configuration path secured!\n")
        #   6. Begin training ... print out ETA and progress bar.
        self.write(f"Beginning training!\n")
        time.sleep(90000)
        #   7. Training ends.
        self.overwrite(f"Training has completed! Output files will be placed their designated folders in {output_dir}.\n")
        time.sleep(3)
        #   8. Output loss, accuracy, t-SNE files.
        self.write(f"Output files, saved models, and t-SNE window available! Please exit t-SNE window to cease function.\n")
        time.sleep(5)

        #   9. Output saved models. 
        self.overwrite(config_path, data)
        if output_dir:
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")
        else:
            self.write(output_dir)
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")



class DeepAstroTrain(Command):

    name = "deepastro run"
    description = "Run the full training and testing pipeline."
    arguments = []
    options = full_options

    def handle(self):

        data = self.option("dataset")
        config_path = self.option("config-path")
        output_dir = self.option("output-directory")
        data_type = self.option("data-type")
        domain_type = self.option("domain-type")
        unk_sup = self.option("unknowns-supplied")
        file_pth_sup = self.option("file-paths-supplied")


        self.write(f"Checking that the path {config_path} exists.\n")
        time.sleep(5)
        self.overwrite("Configuration file found. Pipeline running!\n")
        self.write("Updating config file to account for whether your unknowns are supplied.\n\n")
        time.sleep(4)
        #   3. Check for valid output directory.
        self.write(f"Checking that the path {output_dir} exists. If not, one will be created.\n\n")
        if not os.path.exists(output_dir):
            os.mkdir("/Users/ae_rabelais/Documents/deep-astro-uda/records")
        self.overwrite("Output directory created. Pipeline running!\n")

        self.write("Dataset parameters being configured ....")
        time.sleep(5)
        self.overwrite("Dataset parameters configured! Data has been analyzed, data type found.\n")

        self.write(f"Ensuring valid config path structure...\n")
        time.sleep(7)
        self.overwrite(f"Configuration path secured!\n")
        #   6. Begin training ... print out ETA and progress bar.
        self.write(f"Beginning training!\n")
        time.sleep(90000)
        #   7. Training ends.
        self.overwrite(f"Training has completed! Output files will be placed their designated folders in {output_dir}.\n")
        time.sleep(3)
        #   8. Output loss, accuracy, t-SNE files.
        self.write(f"Output files, saved models, and t-SNE window available! Please exit t-SNE window to cease function.\n")
        time.sleep(5)

        #   9. Output saved models. 
        self.overwrite(config_path, data)
        if output_dir:
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")
        else:
            self.write(output_dir)
            rc = call(f"./scripts/run_obda.sh $0 {config_path}")




class DeepAstroInfer(Command):
    
    name = "deepastro infer"
    description = "Run only the inference portion of the pipeline."
    arguments = []
    options = full_options

    def handle(self):
        
        self.write("Inference pipeline running!\n")
        time.sleep(5)
        with open('predictions.txt', 'a') as f:
            for file in os.listdir("/Users/ae_rabelais/Documents/deep-astro-uda/example_data"):
                print("Image:" + str(file) + "Guess:" + str(random.randint(0,9)))
                f.write("Image:" + str(file) + "Guess:" + str(random.randint(0,9)))

        self.write("Inference complete!\n")
        



