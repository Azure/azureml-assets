# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
File containing function for score.
"""

import os
import json
import argparse
import traceback

import numpy as np

from azureml.train.finetune.core.constants.constants import SaveFileConstants
from azureml.train.finetune.core.drivers.deployment import Deployment
from azureml.train.finetune.core.utils.logging_utils import get_logger_app
from azureml.train.finetune.core.utils.decorators import swallow_all_exceptions
from azureml.train.finetune.core.utils.error_handling.exceptions import ServiceException, ResourceException
from azureml.train.finetune.core.utils.error_handling.error_definitions import DeploymentFailed, PredictionFailed
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

logger = get_logger_app()

DEPLOY_OBJ = None


class _JSONEncoder(json.JSONEncoder):
    """
    custom `JSONEncoder` to make sure float and int64 ar converted
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, datetime.datetime):
        #     return obj.__str__()
        # elif isinstance(obj, Image.Image):
        #     with BytesIO() as out:
        #         obj.save(out, format="PNG")
        #         png_string = out.getvalue()
        #         return base64.b64encode(png_string).decode("utf-8")
        else:
            return super(_JSONEncoder, self).default(obj)


def encode_json(content):
    """
    encodes json with custom `JSONEncoder`
    """
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        cls=_JSONEncoder,
        separators=(",", ":"),
    )


def decode_json(content):
    """
    decode the json content
    """
    return json.loads(content)


@swallow_all_exceptions(logger)
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global DEPLOY_OBJ

    env_var = json.loads(os.environ[SaveFileConstants.DeploymentSaveKey])
    env_var["model_path"] = os.environ.get("AZUREML_MODEL_DIR", None)

    try:
        model_path = env_var["model_path"]
        parent_dir_name = env_var["parent_dir_name"]
        model_parent_dir_name = os.path.basename(model_path)
        if model_parent_dir_name != parent_dir_name:
            model_path = os.path.join(model_path, parent_dir_name)
        env_var["model_path"] = model_path
        logger.info(f"Model path - {model_path}")
        # model directory contents
        for dirpath, _, filenames in os.walk(model_path):
            for filename in filenames:
                logger.info(os.path.join(dirpath, filename))
        args = argparse.Namespace(**env_var)
        logger.info(args)
        DEPLOY_OBJ = Deployment(args)
        # initialize tokenizer and model for prediction
        DEPLOY_OBJ.prepare_prediction_service()
    except Exception as e:
        raise ResourceException._with_error(
            AzureMLError.create(DeploymentFailed, error=e)
        )


@swallow_all_exceptions(logger)
def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    `raw_data` is the raw request body data.
    """
    try:
        data = decode_json(raw_data)
        # pop inputs for pipeline
        inputs = data.pop("inputs", data)
        predictions = DEPLOY_OBJ.predict(inputs)
    except Exception as e:
        # we should never terminate score script by raising exception in run()
        # as it need to continously serve online requests
        # traceback.print_exc()
        predictions = {"msg": "failed", "error": str(e)}
        logger.error(f"Exception: \n", exc_info=True)
    return encode_json(predictions)


'''
if __name__ == '__main__':
    from deploy import get_environment_variables
    os.environ.update(get_environment_variables())
    env_var = json.loads(os.environ[SaveFileConstants.DeploymentSaveKey])
    os.environ['AZUREML_MODEL_DIR'] = env_var["model_checkpoint_dir"]
    init()
    # data = {
    #    # roberta-base
    #    "inputs": [
    #        {"sentence1":"Oh, brother...after hearing about this ridiculous film for umpteen years all \
    #             I can think of is that old Peggy Lee song"},
    #        {"sentence1":"I like the movie"},
    #        {"sentence1":"I hate this genere movie.It is the worst if all time"}
    #    ]
    # }
    #
    # data = {
    #     # bert-base-uncased
    #     "inputs": [
    #         {
    #             "sentence1":"PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .",
    #             "sentence2":"Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So .",
    #             "label":1,
    #             "idx":0
    #         },
    #         {
    #             "sentence1":"The world 's two largest automakers said their U.S. sales declined more than predicted last month as a late summer sales frenzy caused more of an industry backlash than expected .",
    #             "sentence2":"Domestic sales at both GM and No. 2 Ford Motor Co. declined more than predicted as a late summer sales frenzy prompted a larger-than-expected industry backlash .",
    #             "label":1,
    #             "idx":1
    #         }
    #     ]
    # }
    #
    # data = {
    #     # t5-small summarization
    #     "inputs": [
    #         {"document":"The leaflets said the patient had been referred for an urgent appointment as their symptoms might indicate cancer.\nEast Sussex NHS Trust has put the mix-up down to an external company that distributes its printed material.\nIt said the wrong patient information leaflets were added to hospital appointment letters sent out in March.\nIt has now contacted everyone affected to apologise and explain what went wrong.\nLiz Fellows, assistant director of operations at the trust, said: \"It was an administrative error and we apologise for any unnecessary anxiety this error may have caused.\"\nEast Sussex Healthcare NHS Trust covers Hastings, Eastbourne and Rother, and is responsible for the Conquest Hospital and Eastbourne District Hospital.\nThe trust said that due to the large number of appointment letters it sends out it uses an external printing company to print and distribute appointment letters.\nIt said each letter is coded to indicate any supplementary information that needs to accompany it.\nMs Fellows said: \"Unfortunately, for a short period in March, the printing company inadvertently miscoded approximately 850 letters resulting in a 'two-week information leaflet' being inserted with an appointment letter.\n\"As soon as the error became apparent it was stopped immediately, and letters of apology sent out.\"","summary":"Hospital bosses in Sussex have apologised after about 850 patients were sent leaflets in error suggesting they might have cancer.","id":"32672009"},
    #         {"document":"Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday.\nDetectives said three firearms, ammunition and a five-figure sum of money were recovered.\nA 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday.","summary":"A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh.","id":"34227252"},
    #         {"document":"The 48-year-old former Arsenal goalkeeper played for the Royals for four years.\nHe was appointed youth academy director in 2000 and has been director of football since 2003.\nA West Brom statement said: \"He played a key role in the Championship club twice winning promotion to the Premier League in 2006 and 2012.\"","summary":"West Brom have appointed Nicky Hammond as technical director, ending his 20-year association with Reading.","id":"36175342"}
    #     ]
    # }
    #
    # data = {
    #     # ts-small translation
    #     "inputs": [
    #         {"en":"Others have dismissed him as a joke.","ro":"Al\u021bii l-au numit o glum\u0103."},
    #         {"en":"Marco Rubio with only 21 percent.","ro":"Marco Rubio, cu doar 21%."},
    #         {"en":"He's in a dead heat with Ted Cruz.","ro":"Este aproape la egalitate cu Ted Cruz."}
    #     ]
    # }


    print(run(encode_json(data)))
    # for _ in range(10):
    #     print(run(encode_json(data)))
    #     print("#"*150)
'''
