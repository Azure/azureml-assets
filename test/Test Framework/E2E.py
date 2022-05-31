# %%
from pathlib import Path
import os

# %%
DIR=Path("../../vision/src/pipelines/canary/classification_random.yml")
stream = os.popen("az ml job create -f {DIR}")
output = stream.read()

# %%
print(output)



# %%
