# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
from typing import Tuple

import click
from azureml.core import (ComputeTarget, Dataset, Environment,
                          RunConfiguration, Workspace)
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.experiment import Experiment
from azureml.pipeline.core import (Pipeline, PipelineData, PipelineParameter,
                                   PublishedPipeline)
from azureml.pipeline.steps import DatabricksStep, PythonScriptStep

CLI_AUTH = AzureCliAuthentication()
# noinspection PyTypeChecker
WS = Workspace.from_config(auth=CLI_AUTH)
RC = RunConfiguration()
RC.environment = Environment.get(WS, "lightgbm")


# noinspection PyTypeChecker
def create_databricks_step(
        input_dataset: Dataset,
        compute: ComputeTarget,
        debug_run: bool
) -> Tuple[DatabricksStep, PipelineData]:
    output_data = PipelineData(
        name="ParquetFiles",
        datastore=WS.get_default_datastore(),
        is_directory=True
    )

    node_size = 'Standard_DS4_v2'
    spark_version = '7.3.x-cpu-ml-scala2.12'

    db_step = DatabricksStep(
        name='Convert to Parquet',
        inputs=[input_dataset.as_named_input("CSVFiles")],
        outputs=[output_data],
        source_directory="./safe-driver/prep_data",
        python_script_name='prep_data.py',
        python_script_params=["--number-of-files", "1"],  # Set the number of output files to 1
        num_workers=1,
        compute_target=compute,
        pypi_libraries=[],
        allow_reuse=debug_run,
        node_type=node_size,
        spark_version=spark_version,
    )

    return db_step, output_data


# noinspection PyTypeChecker
def create_train_model_step(
        input_data: PipelineData,
        compute: ComputeTarget,
        debug_run: bool
) -> Tuple[PythonScriptStep, PipelineData]:
    output_folder = "./outputs"
    output_data = PipelineData(
        name="ModelMetadata",
        datastore=WS.get_default_datastore(),
        is_directory=True, output_path_on_compute=output_folder, output_mode="upload"
    )

    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        source_directory='./safe-driver/train/',
        compute_target=compute,
        inputs=[input_data],
        outputs=[output_data],
        allow_reuse=debug_run,
        arguments=["--output-folder", output_folder, "--training-data", input_data.as_mount()],
        runconfig=RC
    )

    return train_step, output_data


# noinspection PyTypeChecker
def create_evaluate_model_step(
        model_metadata_folder: PipelineData,
        compute: ComputeTarget,
        validation_data: Dataset,
        debug_run: bool) -> Tuple[PythonScriptStep, PipelineData]:
    """
    Creates "Evaluate Model" Step
    """
    output_folder = "./outputs"
    output_data = PipelineData(
        name="RegisterModel",
        datastore=WS.get_default_datastore(),
        is_directory=True,
        output_path_on_compute=output_folder,
        output_mode="upload"
    )

    eval_step = PythonScriptStep(
        name="Evaluate Model",
        script_name="evaluate.py",
        source_directory='./safe-driver/evaluate/',
        compute_target=compute,
        inputs=[model_metadata_folder],
        outputs=[output_data],
        arguments=[
            "--model-metadata", model_metadata_folder.as_mount(),
            "--register-model-folder", output_folder,
            "--validation-data", validation_data.as_named_input("ValidationData").as_mount()
        ],
        allow_reuse=debug_run,
        runconfig=RC
    )

    return eval_step, output_data


# noinspection PyTypeChecker
def create_register_model_step(
        model_folder: PipelineData,
        register_model_folder: PipelineData,
        compute: ComputeTarget,
        debug_run: bool
) -> PythonScriptStep:
    """
    Creates "Register Model" PythonScriptStep
    """
    force_param = PipelineParameter(name="force_registration", default_value="False")
    skip_param = PipelineParameter(name="skip_registration", default_value="False")

    register_step = PythonScriptStep(
        name="Register Model",
        script_name="register_model.py",
        source_directory='./safe-driver/register/',
        compute_target=compute,
        inputs=[
            model_folder,
            register_model_folder
        ],
        arguments=[
            "--force", force_param,
            "--skip", skip_param,
            "--model-metadata", model_folder.as_mount(),
            "--register-model-folder", register_model_folder.as_mount()
        ],
        allow_reuse=debug_run,
        runconfig=RC
    )

    return register_step


def create_pipeline(
        debug_run: bool,
        dbx_compute: str,
        aml_compute: str,
        input_dataset: str,
        validation_dataset: str
) -> Pipeline:
    """
    Creates the overall Pipeline

    Dataset -> Convert to Parquet -> Create Features -> (Train Model -> Evaluate Model) -> Register Model
    """
    cpu_cluster = WS.compute_targets[aml_compute]
    databricks_cluster = WS.compute_targets[dbx_compute]

    raw_csv_data = WS.datasets[input_dataset]
    validation_data = WS.datasets[validation_dataset]

    # Convert to Parquet Step
    db_step, parquet_data = create_databricks_step(
        input_dataset=raw_csv_data,
        compute=databricks_cluster,
        debug_run=debug_run
    )

    # Train Model Step
    train_step, model_folder = create_train_model_step(
        input_data=parquet_data,
        compute=cpu_cluster,
        debug_run=debug_run
    )

    # Evaluate Model Step
    eval_step, register_model_folder = create_evaluate_model_step(
        model_metadata_folder=model_folder,
        compute=cpu_cluster,
        validation_data=validation_data,
        debug_run=debug_run
    )

    # Register Model Step
    reg_step = create_register_model_step(
        model_folder=model_folder,
        register_model_folder=register_model_folder,
        compute=cpu_cluster,
        debug_run=debug_run
    )

    return Pipeline(WS, steps=[reg_step])


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--force-model-register", "-f", is_flag=True,
              help="Force the model registration. Ignores model performance against the existing model")
@click.option("--skip-model-register", "-k", is_flag=True,
              help="Skip the model registration. Ignores model performance against the existing model")
@click.option("--submit-pipeline", "-s", is_flag=True,
              help="Submit the pipeline for a run")
@click.option("--publish-pipeline", "-p", is_flag=True,
              help="Publish the pipeline")
@click.option("--experiment-name", type=str, default="pipeline-test",
              help="If submitting pipeline, submit under this experiment name")
@click.option("--databricks-compute-name", "dbx_cluster_name", type=str, default='adb-mlops',
              help="The name of the compute target to run Databricks steps against")
@click.option("--aml-compute-name", type=str, default='cpu-cluster',
              help="The name of the compute target to run Python Script steps against")
@click.option("--input-dataset-name", type=str, default="safe_driver_csv",
              help="The name of the input dataset")
@click.option("--validation-dataset-name", type=str, default="safe_driver_validation")
@click.option("--debug-run", is_flag=True, default=False)
def main(force_model_register: bool,
         skip_model_register: bool,
         submit_pipeline: bool,
         publish_pipeline: bool,
         experiment_name: str,
         debug_run: bool,
         dbx_cluster_name: str,
         aml_compute_name: str,
         input_dataset_name: str,
         validation_dataset_name: str):
    pipeline: Pipeline = create_pipeline(
        debug_run=debug_run,
        dbx_compute=dbx_cluster_name,
        aml_compute=aml_compute_name,
        input_dataset=input_dataset_name,
        validation_dataset=validation_dataset_name
    )
    pipeline.validate()

    if submit_pipeline and not publish_pipeline:
        exp = Experiment(WS, experiment_name)
        exp.submit(pipeline, pipeline_parameters={"force_registration": str(force_model_register),
                                                  "skip_registration": str(skip_model_register)})

    if publish_pipeline:
        published_pipeline: PublishedPipeline = pipeline.publish(
            name="Driver Safety Pipeline",
            description="Training Pipeline for new driver safety model"
        )

        if submit_pipeline:
            published_pipeline.submit(
                workspace=WS,
                experiment_name=experiment_name,
                pipeline_parameters={"force_registration": str(force_model_register),
                                     "skip_registration": str(skip_model_register)}
            )

        sys.stdout.write(published_pipeline.id)


if __name__ == "__main__":
    main()
