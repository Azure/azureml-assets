// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using OpenAI.Files;
using OpenAI.VectorStores;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with File Search tool.
/// Creates a vector store, uploads a product catalog document,
/// then evaluates the agent's ability to answer questions about it.
/// </summary>
public class FileSearchEvaluationTests : AgentEvaluatorTestBase
{
    public FileSearchEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithFileSearch()
    {
        var assetDir = Path.Combine(AppContext.BaseDirectory, "assets");
        var productFile = Path.Combine(assetDir, "sample_product_info.txt");

        var fileClient = ProjectClient.OpenAI.GetOpenAIFileClient();
        var vectorStoreClient = ProjectClient.OpenAI.GetVectorStoreClient();

        // Upload file and create vector store
        var uploadResult = fileClient.UploadFile(productFile, FileUploadPurpose.Assistants);
        var vectorStore = vectorStoreClient.CreateVectorStore(
            new VectorStoreCreationOptions { Name = UniqueName("E2E-VectorStore") }
        );
        vectorStoreClient.AddFileToVectorStore(vectorStore.Value.Id, uploadResult.Value.Id);

        var agentName = UniqueName("E2E-FileSearch");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful assistant. Search the uploaded files to answer product questions.",
            new { type = "file_search", vector_store_ids = new[] { vectorStore.Value.Id } }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "What is the price and battery life of the Contoso SmartWatch X100?",
                $"File Search E2E - {agentName}"
            );
            AssertEvaluationResults(outputItems);
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
            vectorStoreClient.DeleteVectorStore(vectorStore.Value.Id);
        }
    }
}
