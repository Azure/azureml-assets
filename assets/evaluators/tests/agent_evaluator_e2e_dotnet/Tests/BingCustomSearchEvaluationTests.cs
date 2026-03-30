// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Bing Custom Search tool.
/// Requires: BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID, BING_CUSTOM_SEARCH_INSTANCE_NAME
/// </summary>
public class BingCustomSearchEvaluationTests : AgentEvaluatorTestBase
{
    public BingCustomSearchEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithBingCustomSearch()
    {
        if (!HasEnv("BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID", "BING_CUSTOM_SEARCH_INSTANCE_NAME"))
            throw new InvalidOperationException("Missing environment variables: BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID, BING_CUSTOM_SEARCH_INSTANCE_NAME");

        var connectionId = RequireEnv("BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID");
        var instanceName = RequireEnv("BING_CUSTOM_SEARCH_INSTANCE_NAME");

        var tool = new BingCustomSearchPreviewTool(
            new BingCustomSearchToolParameters(
                new[] { new BingCustomSearchConfiguration(connectionId, instanceName) }
            )
        );

        var agentName = UniqueName("E2E-BingCustomSearch");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant. Use Bing Custom Search to find relevant information from curated websites.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "Search for the latest product announcements.",
                $"Bing Custom Search E2E - {agentName}"
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
