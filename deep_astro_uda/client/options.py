from cleo import option

config_path = option(
        long_name="config-path", 
        description="The path to the configuration file used to define the training, inference, and output regimen.",
        value_required=False
)

# TODO: Add descriptions.
full_options = [
        option(long_name="dataset", value_required=True),
        option(long_name="config-path", value_required=False),
        option(long_name="unknowns-supplied", value_required=True),
        option(long_name="image-path-text", value_required=False),
        option(long_name="output-directory", value_required=False),
        option(long_name="data-type", value_required=False),
        option(long_name="domain-type", value_required=False),  # implement later
    ]