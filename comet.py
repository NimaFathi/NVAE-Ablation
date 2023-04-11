import comet_ml
import random


class CometML:
    def __init__(self, api_key, project_name, workspace, disabled, parameters) -> None:
        self.api_key = api_key
        self.project_name = project_name
        self.parameters = parameters
        self.experiment = comet_ml.Experiment(api_key=self.api_key, disabled=disabled, project_name=self.project_name,
                                              workspace=workspace, auto_metric_logging=True, log_env_details=True,
                                              log_env_host=True, log_code=True)
        self.experiment.log_parameters(self.parameters)
        self.experiment.add_tag(f"{self.parameters['id']}")

    def get_experiment(self):
        return self.experiment
