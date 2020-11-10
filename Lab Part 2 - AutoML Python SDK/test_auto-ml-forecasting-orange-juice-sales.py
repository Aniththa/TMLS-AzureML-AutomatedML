import sys

from checknotebookoutput import checkNotebookOutput
from checkexperimentresult import checkExperimentResult
from checkexperimentresult import check_experiment_model_explanation_of_best_run
from download_run_files import download_run_files

# We need to download files for the remote run. Since the following check methods would raise exceptions.
download_run_files(experiment_names=['automl-ojforecasting'], download_all_runs=True)

checkExperimentResult(experiment_name='automl-ojforecasting',
                      expected_num_iteration='1000',
                      expected_minimum_score=0.01,
                      expected_maximum_score=0.3,
                      metric_name='normalized_root_mean_squared_error',
                      absolute_minimum_score=0.0,
                      absolute_maximum_score=1.0)

check_experiment_model_explanation_of_best_run(experiment_name='automl-ojforecasting')

# Check the output cells of the notebook.
# We need to suppress this warning '[except]Warning, azureml-defaults not detected' since it's popped
# from the Aci deploy_configuration(), as the azureml-defaults is a required package by Aci.
# This won't impact the customer and it's not critical.
# [stderr] checks for any messages written to stderr, typically logger.warning() will output this.

allowed_warn_str = ('[except]warning - retrying'
                    '[except]UserWarning: Matplotlib is building the font cache'
                    '[except]warning: a newer version of conda exists'
                    '[except]Warning, azureml-defaults not detected'
                    '[except]UserWarning: Starting from version 2.2.1, '
                    'the library file in distribution wheels for macOS is built by the Apple Clang'
                    '[except]brew install libomp'
                    '[except]Using different time series parameters in AutoML configs'
                    '[except]Forecasting parameter country_or_region will be deprecated in the future,'
                    '[except]Forecasting parameter max_horizon will be deprecated in the future,'
                    '[except]reg:linear is now deprecated'
                    '[except]Forecasting parameter grain_column_names will be deprecated in the future')

checkNotebookOutput('auto-ml-forecasting-orange-juice-sales.nbconvert.ipynb' if len(sys.argv) < 2 else sys.argv[1],
                    'warning{}'.format(allowed_warn_str),
                    '[stderr]{}{}'.format(allowed_warn_str,
                                          '[except]Importing plotly failed'))
