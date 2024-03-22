import mlflow

import pandas as pd
import base64
test_image_paths = ["/home/azureuser/workspace/AzureMlCli/src/azureml-acft-image-components/tests/tests_data/images/od_is_images/10.jpg"]

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


test_df = pd.DataFrame(
    data=[
        base64.encodebytes(read_image(image_path)).decode("utf-8")
        for image_path in test_image_paths
    ],
    columns=["image"],
)

model = mlflow.pyfunc.load_model("./output")
resp = model.predict(test_df, params={"text_prompt" : "can. "})
print(resp)
aa = 1