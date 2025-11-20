# Running Unit Tests Locally
In order to run unit tests locally, first ensure you have the necessary requirements installed defined by requirements.txt in the tests directory. If not, the follow the steps below. . Note that not all tests will be available. Any test that imports mii or vllm will not be able to run on a local Windows OS as they are not compatible with Windows. In order to run all tests, check out a Compute Instance and install the necessary packages there.
1. cd into the tests directory
2. Run pip install -r requirements.txt in the conda environment. This will install the necessary requirements to run pytest
3. If you enter a test file such as test_conversation.py and see a yellow squiggly line underneath *conversation* in the line *from conversation import ....* hover on the word *conversation*. There will be an option that pops up that says View Problem and Quick Fix. Select Quick Fix and select *Add ./fm-inference/src to ExtraPaths*.
4. If you have trouble with step 3, instead go to the .vscode folder. Inside there should be a settings.json file (if not create one). If you create one then copy: 

        {
            "python.linting.flake8Args": [ 
                "--max-line-length=119", 
                "--ignore=W503", 
            ],
            "python.testing.unittestEnabled": false,
            "python.testing.pytestEnabled": true,
            "python.analysis.extraPaths": [
                "./fm-inference/src"
            ]
        }

    If the directory already has a settings.json file, simply add 

            "python.analysis.extraPaths": [
                "./fm-inference/src"
            ]

    to the file.

5. Now open a vscode command line terminal and activate your conda environment. Run pytest *PATH_TO_TEST_FILE* and the unit test should run

## Optionally
To use the builtin VSCode testing, do the following
1. Follow steps 1-4 from above
2. Once that is completed, Click *Ctrl + Shift + P*. A search bar should appear
3. Type in *Configure Tests*, then select *pytest*, and then *fm-inference*. This should start the test discovery process. 
4. Click on the test flask icon located on the left panel in VSCode. If test discovery was successful, a dropdown menu will be available. Keep opening the dropdown menus until you reach the unit test file you like then click the play icon.


### Failures

#### Conda Environment
If there is a failure, also ensure you have the correct conda environment selected. To change this, you can click *Ctrl + Shift + P* and search *Select Interpreter*. Then choose the Python version that is associated with your conda environment. Then rerun *Configure Tests*.

#### Test Discovery
If test discovery was successful. A dropdown menu will appear with foundation-models-inference at the top, followedby fm-inference, then tests. You can see this if you click the arrows to the left hand side of the titles. Clicking the play icon for the foundation-models-inference menu In order to actually run tests, you must click the play icon on the fm-inference menu or any submenu below it.