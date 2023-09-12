import base64

import mlflow.pyfunc
import pandas as pd


if __name__ == "__main__":
    pyfunc_model = mlflow.pyfunc.load_model("mlflow_model_folder")

    # image
    image = "https://llava-vl.github.io/static/images/view.jpg"
    # image = base64.encodebytes(open("/home/azureuser/cloudfiles/code/Users/rdondera/testing/VizWiz_val_00000014.jpg", "rb").read()).decode("utf-8")

    # prompt (note: copying what comes from conversation class)
    prompt_template_mpt1 = \
        "<|im_start|>system" + "\n" + \
        "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user" + "\n" + \
        "<im_start><image><im_end>" + "\n" + \
        "{user_input}<|im_end|><|im_start|>assistant" + "\n"
    prompt_mpt2 = ""

    prompt = prompt_template_mpt1.format(user_input="What's in this image?")

    test_df = test_df = pd.DataFrame(data=[[image, prompt]], columns=["image", "prompt"])

    result = pyfunc_model.predict(test_df).to_json(orient="records")
    print("testing result:", result)
