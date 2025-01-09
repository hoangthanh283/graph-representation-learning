import os

import neptune

NEPTUNE_PRJ_NAME = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
if NEPTUNE_PRJ_NAME and NEPTUNE_API_TOKEN:
    NEPTUNE_RUN = neptune.init_run(project=NEPTUNE_PRJ_NAME, api_token=NEPTUNE_API_TOKEN)
else:
    NEPTUNE_RUN = None
