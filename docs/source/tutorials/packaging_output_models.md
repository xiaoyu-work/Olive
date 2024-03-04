# Packaing Olive artifacts

## What is Olive Packaging
Olive will output multiple candidate models based on metrics priority ranks. It also can package output artifacts when the user required. Olive packaging can be used in different scenarios. There are two packaging types: `Zipfile` and `AzureML`.


### Zipfile
Zipfile packaging will generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and `ONNXRuntimePackages` in the `output_dir` folder (from Engine Configuration):
* `CandidateModels`: top ranked output model set
    * Model file
    * Olive Pass run history configurations for candidate model
    * Inference settings (`onnx` model only)
* `SampleCode`: code sample for ONNX model
    * C++
    * C#
    * Python
* `ONNXRuntimePackages`: ONNXRuntime package files with the same version that were used by Olive Engine in this workflow run.

#### CandidateModels
`CandidateModels` includes k folders where k is the number of output models, with name `BestCandidateModel_1`, `BestCandidateModel_2`, ... and `BestCandidateModel_k`. The order is ranked by metrics priorities. e.g., if you have 3 metrics `metric_1`, `metric_2` and `metric_3` with priority rank `1`, `2` and `3`. The output models will be sorted firstly by `metric_1`. If the value of `metric_1` of 2 output models are same, they will be sorted by `metric_2`, and followed by next lower priority metric.

Each `BestCandidateModel` folder will include model file/folder. A json file which includes the Olive Pass run history configurations since input model. And a json file for inference settings for the candidate model if the candidate model is an ONNX model.

#### SampleCode
Olive will only provide sample codes for ONNX model. Sample code supports 3 different programming languages: `C++`, `C#` and `Python`. And a code snippet introducing how to use Olive output artifacts to inference candidate model with recommended inference configurations.

### AzureML
AzureML packaging will create an Environment in you Azure ML workspace with Olive top ranked No.1 output model and scoring script. You can also make a deployment to your endpoint with this environment.

## How to package Olive artifacts
Olive packaging configuration is configured in `PackagingConfig` in Engine configuration. If not specified, Olive will not package artifacts.

* `PackagingConfig`
    * `type [PackagingType]`:
      Olive packaging type. Olive will package different artifacts based on `type`.
    * `name [str]`:
      For `PackagingType.Zipfile` type, Olive will generate a ZIP file with `name` prefix: `<name>.zip`. By default, the output artifacts will be named as `OutputModels.zip`.
    * `azureml_config [AzureMLPackagingConfig]`:
      The configurations for Azure ML type packaging

* `AzureMLPackagingConfig`
    * `model_package [ModelPackageConfig]`:
      The configurations for model packaging.
    * `model_name [str]`:
      The model name when registering your output model to your Azure ML workspace.
    * `model_version [str]`:
      The model version when registering your output model to your Azure ML workspace. Please note if there is already a model with the same name and the same version in your workspace, this will override your existing registered model.
    * `deployment_config [DeploymentConfig]`
      The deployment configuration. If not specified, Olive will not make the delopment.

* `ModelPackageConfig`
    * `target_environment_name [str]`:
      The environment name for the environment created by Olive.
    * `target_environment_version [str]`
      The environment version for the environment created by Olive. Please note if there is already an environment with the same name and the same version in your workspace, your existing environment version will plus 1. This `target_environment_version` will not be applied for your environment.
    * `inferencing_server [InferenceServerConfig]`
      * `type [str]`
        The targeted inferencing server type. `AzureMLOnline` or `AzureMLBatch`.
      * `code_folder [str]`
        The folder path to your scoring script.
      * `scoring_script [str]`
            The scoring script name.
    * `base_environent_id [str]`
      The base environment id that will be used for Azure ML packaging. The format is `azureml:<base-environment-name>:<base-environment-version>`.
    * `environment_variables [dict]`
      Env vars that are required for the package to run, but not necessarily known at Environment creation time.
    * `tags [dict]`
      Tags to be included in the object

* `DeploymentConfig`
    * `endpoint_name [str]`
      The endpoint name for the deployment. If the endpoint doesn't exist, Olive will create one endpoint with this name.
    * `deployment_name [str]`
      The name of the deployment.
    * `instance_type [str]`
      Azure compute sku. ManagedOnlineDeployment only. Default to Standard_DS3_v2
    * `compute [str]`
      Compute target for batch inference operation. BatchDeployment only.
    * `instance_count [str]`
      Number of instances the interfering will run on, default to 1.
    * `mini_batch_size [str]`
      Size of the mini-batch passed to each batch invocation, default to 10.
    * `extra_config [dict]`
      Extra configurations for deployment. Find more details in [ManagedOnlineDeployment]() and [BatchDeployment]()


You can add `PackagingConfig` to Engine configurations. e.g.:

Zipfile example:
```
"engine": {
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe",
        "search_algorithm_config": {
            "num_samples": 5,
            "seed": 0
        }
    },
    "evaluator": "common_evaluator",
    "host": "local_system",
    "target": "local_system",
    "packaging_config": {
        "type": "Zipfile",
        "name": "OutputModels"
    },
    "clean_cache": true,
    "cache_dir": "cache"
}
```

AzureML example:
```
"engine": {
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe",
        "search_algorithm_config": {
            "num_samples": 5,
            "seed": 0
        }
    },
    "packaging_config": {
        "type": "AzureML",
        "azureml_config": {
            "model_package": {
                "target_environment_name": "olive_target_environment",
                "target_environment_version": "1",
                "inferencing_server": {
                    "type": "AzureMLOnline",
                    "code_folder": "code",
                    "scoring_script": "score.py"
                },
                "base_environent_id": "azureml:aml-packaging:1"
            },
            "model_name": "olive_model_name",
            "model_version": "1",
            "deployment_config": {
                "endpoint_name": "endpoint-name",
                "deployment_name": "deployment-name",
                "instance_type": "Standard_DS3_v2",
                "instance_count": 1
            }
        }
    },
    "evaluator": "common_evaluator",
    "host": "local_system",
    "clean_cache": true,
    "cache_dir": "cache"
}
```