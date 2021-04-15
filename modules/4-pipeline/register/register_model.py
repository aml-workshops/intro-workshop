# Copyright (c) 2021 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
from pathlib import Path
from typing import Tuple
import logging

import click
from azureml.core import Run, Experiment, Workspace

# Retrieve the run, experiment and workspace
run = Run.get_context()
experiment: Experiment = run.experiment
workspace: Workspace = experiment.workspace

METADATA_JSON = "metadata.json"
RECOMMENDATION_JSON = 'recommend.json'

def read_metadata(folder_path: str) -> dict:
    model_metadata_path = Path(folder_path)
    model_metadata_file = model_metadata_path / METADATA_JSON

    with open(model_metadata_file, 'r') as fo:
        metadata = json.load(fo)

    return metadata

def read_recommendation(folder_path: str) -> bool:
    recommendation_path = Path(folder_path)
    recommendation_file = recommendation_path / RECOMMENDATION_JSON

    with open(recommendation_file, 'r') as fo:
        recommendation = json.load(fo)

    return recommendation.get('register', False)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--force", type=bool, default=False,
              help="Force model registration")
@click.option("--skip", type=bool, default=False,
              help="Skip model registration")
@click.option("--model-metadata", required=True,
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=True),
              help="The folder where the model files are stored")
@click.option("--register-model-folder", "register_folder", required=True,
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=True),
              help="The folder where the deploy indicator is stored")
@click.option("--model-name", type=str)
def main(force: bool, skip: bool, model_metadata: str, register_folder: str, model_name: str):
    if force and skip:
        raise ValueError("Model registration cannot be both forced and skipped")
    
    # Determine if the model should be registered
    if skip:
        print("Registration skipped")
        register_recommended = False
    elif force:
        print("Model registration forced")
        register_recommended = True
    else:
        register_recommended = read_recommendation(folder_path=register_folder)
        print(f"Model Registration Is Recommended?: {register_recommended}")
    

    # If model registration is recommended, then register the model
    if register_recommended:
        metadata = read_metadata(model_metadata)
        challenger_model_run = Run.get(workspace=workspace, run_id=metadata['run_id'])
        
        challenger_model_run.register_model(
            model_name, 
            model_path=metadata['model_path']
            )


if __name__ == "__main__":
    main()
