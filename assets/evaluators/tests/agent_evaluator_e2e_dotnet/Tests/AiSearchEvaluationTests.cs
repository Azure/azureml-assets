// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Azure AI Search tool.
/// Requires: AI_SEARCH_PROJECT_CONNECTION_ID, AI_SEARCH_INDEX_NAME
/// </summary>
public class AiSearchEvaluationTests : AgentEvaluatorTestBase
{
    public AiSearchEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithAiSearch()
    {
        if (!HasEnv("AI_SEARCH_PROJECT_CONNECTION_ID", "AI_SEARCH_INDEX_NAME"))
            throw new InvalidOperationException("Missing environment variables: AI_SEARCH_PROJECT_CONNECTION_ID, AI_SEARCH_INDEX_NAME");

        var connectionId = RequireEnv("AI_SEARCH_PROJECT_CONNECTION_ID");
        var indexName = RequireEnv("AI_SEARCH_INDEX_NAME");

        var index = new AzureAISearchToolIndex()
        {
            ProjectConnectionId = connectionId,
            IndexName = indexName,
            QueryType = AzureAISearchQueryType.Simple,
        };
        var tool = new AzureAISearchTool(
            new AzureAISearchToolOptions(new[] { index })
        );

        var agentName = UniqueName("E2E-AISearch");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant. Use the search tool to find relevant information. Cite sources when possible.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "Search for the most relevant information available in the index.",
                $"AI Search E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: UnsupportedToolEvaluators
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
