from cleo import option

demo_options = [
        option(long_name="output-directory", value_required=False),
        option(long_name="domain-type", value_required=True),
        option(long_name="config-path", value_required=False),
        option(long_name="dataset", value_required=True),
        ]

full_options = [
        option(long_name="dataset", value_required=True),
        option(long_name="config-path", value_required=False),
        option(long_name="unknowns-supplied", value_required=True),
        option(long_name="file-paths-supplied", value_required=False), #change to image-path-text
        option(long_name="output-directory", value_required=False),
        option(long_name="data-type", value_required=False),
        option(long_name="domain-type", value_required=False),  # implement later
        # add data type option
    ]