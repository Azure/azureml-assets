// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.Net.Http.Headers;
using System.Text.Json;
using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Azure.Core;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Memory Search tool.
/// Seeds memory via a first conversation, waits for extraction,
/// then evaluates recall in a second conversation.
/// Requires: MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME, MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME
/// </summary>
public class MemorySearchEvaluationTests : AgentEvaluatorTestBase
{
    public MemorySearchEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithMemorySearch()
    {
        if (!HasEnv("MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME", "MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME"))
            throw new InvalidOperationException("Missing environment variables: MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME, MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME");

        var memoryStoreName = GetEnv("MEMORY_STORE_NAME") ?? "my_memory_store";
        var userScope = UniqueName("e2e_user");

        var tool = new MemorySearchPreviewTool(memoryStoreName, userScope) { UpdateDelay = 1 };

        var agentName = UniqueName("E2E-MemorySearch");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant with memory. Remember what the user tells you.",
                    Tools = { tool },
                }
            )
        );

        string? conv1Id = null;
        string? conv2Id = null;
        try
        {
            // First conversation – seed memory
            conv1Id = CreateConversation("My favorite color is blue and I live in Portland, Oregon.");
            CreateResponseForAgentWithConversation(agent.Value.Name, conv1Id);

            // Wait for memory extraction
            Output.WriteLine("Waiting 60s for memory extraction...");
            Thread.Sleep(60_000);

            // Second conversation – test recall
            conv2Id = CreateConversation("What is my favorite color and where do I live?");
            var responseId = CreateResponseForAgentWithConversation(agent.Value.Name, conv2Id);

            // Evaluate the recall response
            var (evalId, runId, outputItems) = RunEvaluationWithResponseId(
                responseId,
                agent.Value.Name,
                $"Memory Search E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems
            );
        }
        finally
        {
            DeleteConversation(conv1Id);
            DeleteConversation(conv2Id);
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
