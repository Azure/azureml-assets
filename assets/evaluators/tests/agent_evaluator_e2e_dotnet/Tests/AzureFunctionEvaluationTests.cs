// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Azure Function tool.
/// Requires: STORAGE_INPUT_QUEUE_NAME, STORAGE_OUTPUT_QUEUE_NAME, STORAGE_QUEUE_SERVICE_ENDPOINT
/// </summary>
public class AzureFunctionEvaluationTests : AgentEvaluatorTestBase
{
    public AzureFunctionEvaluationTests(ITestOutputHelper output) : base(output) { }

    [Fact]
    public void EvaluateAgentWithAzureFunction()
    {
        if (!HasEnv("STORAGE_INPUT_QUEUE_NAME", "STORAGE_OUTPUT_QUEUE_NAME", "STORAGE_QUEUE_SERVICE_ENDPOINT"))
            throw new InvalidOperationException("Missing environment variables: STORAGE_INPUT_QUEUE_NAME, STORAGE_OUTPUT_QUEUE_NAME, STORAGE_QUEUE_SERVICE_ENDPOINT");

        var inputQueueName = RequireEnv("STORAGE_INPUT_QUEUE_NAME");
        var outputQueueName = RequireEnv("STORAGE_OUTPUT_QUEUE_NAME");
        var queueEndpoint = RequireEnv("STORAGE_QUEUE_SERVICE_ENDPOINT");

        var parameters = BinaryData.FromObjectAsJson(new
        {
            type = "object",
            properties = new
            {
                location = new
                {
                    type = "string",
                    description = "City or location to get weather for",
                },
            },
        });

        var function = new AzureFunctionDefinitionFunction("queue_trigger", parameters)
        {
            Description = "Get weather for a given location",
        };
        var inputBinding = new AzureFunctionBinding(new AzureFunctionStorageQueue(inputQueueName, queueEndpoint));
        var outputBinding = new AzureFunctionBinding(new AzureFunctionStorageQueue(outputQueueName, queueEndpoint));

        var tool = new AzureFunctionTool(
            new AzureFunctionDefinition(function, inputBinding, outputBinding)
        );

        var agentName = UniqueName("E2E-AzureFunction");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant. Use the Azure Function tool to get weather information.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "What is the weather in New York City?",
                $"Azure Function E2E - {agentName}"
            );
            AssertEvaluationResults(outputItems);
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
