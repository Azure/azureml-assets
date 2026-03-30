// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Bing Grounding tool.
/// Requires: BING_PROJECT_CONNECTION_ID
/// </summary>
public class BingGroundingEvaluationTests : AgentEvaluatorTestBase
{
    public BingGroundingEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithBingGrounding()
    {
        if (!HasEnv("BING_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variable: BING_PROJECT_CONNECTION_ID");

        var connectionId = RequireEnv("BING_PROJECT_CONNECTION_ID");

        var tool = new BingGroundingTool(
            new BingGroundingSearchToolOptions(
                new[] { new BingGroundingSearchConfiguration(connectionId) }
            )
        );

        var agentName = UniqueName("E2E-BingGrounding");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant. Use Bing search for grounding your answers with current information.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "What are the latest developments in quantum computing?",
                $"Bing Grounding E2E - {agentName}"
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
