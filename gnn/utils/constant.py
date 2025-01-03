import os

import neptune

NEPTUNE_RUN = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)
