This is a flow demonstrating Q&A with GPT3.5 using data from your own indexed files to make the answer more grounded. It involves embedding the raw query, using "Vector Search" to find most relevant context in user data, and then using GPT3.5 to generate an answer to the question with the documents. This example also contains multiple prompt variants that you can tune.

Brief description: Create flows for Q&A with GPT3.5 using data from your own indexed files to make the answer more grounded for entreprise chat scenarios.

### What you will learn

In this flow, you will learn

* how to compose a QA system flow.
* how to use Vector Search Tool to find relevant documents and leverage domain knowledge.
* how to tune prompt variants.

### Getting Started

#### 1 Create Azure OpenAI or OpenAI connection
Go to Prompt Flow "Connection" tab. Click on "Add" button, and start to set up your "AzureOpenAI" or "OpenAI" connection.

#### 2 Create Vector Index(Optional)
This is an optional step if you want a sample with your own data. It might take some time to create a vector index from your data (depending on your data size), so if you do not wish to wait, you can skip this step and move on to step 3.

Below the page there is a button **"Create Vector Index"**. You can click the button to create a vector index for your own data with a UI wizard. After the vector index gets created, a Prompt flow will automatically be created afterwards. You should be able to jump to the flow for your data in vector index UI.

You can skip the rest of the steps below if you choose this option.

#### 3 Prepare bulkTest data

Download example bulk test data from <a href="https://ragsample.blob.core.windows.net/ragdata/QAGenerationData.jsonl" target="_blank">here</a> to your local device. Later you will use it when submit bulk test run.

#### 4 Configure the flow with your connection
Create or clone a new flow, go to the step need to configure connection. Select your connection from the connection drop-down box and fill in the configurations.

#### 5 Configure mlindex data
Go to the step "search_question_from_indexed_docs", and in the "path" field, there is a pre-filled uri that contains example data, you can try it. If you want to try your own data, use "Create Vector Index" button from step 2.

#### 6 Submit and Run with Bulk Test
Type in your question. There is pre-filled questions in the example flow, but you can try any questions you want. Click "Run" button on the flow page. The flow will be executed. 
Beside "Run" button, there is a "Bulk Test" button. Click on it and you will be landed to configuration page. In "Bulk test settings" page, there is a **"+upload new data"** button, click and upload the file you downloaded from Step 3. You should be able to run bulk test runs with settings.


### Tools used in this flow

* LLM Tool
* Vector Index Lookup Tool
* Python Tool