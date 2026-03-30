// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with SharePoint tool.
/// Requires: SHAREPOINT_PROJECT_CONNECTION_ID
/// </summary>
public class SharepointEvaluationTests : AgentEvaluatorTestBase
{
    public SharepointEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithSharepoint()
    {
        if (!HasEnv("SHAREPOINT_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variable: SHAREPOINT_PROJECT_CONNECTION_ID");

        var connectionId = RequireEnv("SHAREPOINT_PROJECT_CONNECTION_ID");

        var tool = new SharepointPreviewTool(
            new SharePointGroundingToolOptions
            {
                ProjectConnections = { new ToolProjectConnection(connectionId) },
            }
        );

        var agentName = UniqueName("E2E-SharePoint");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant. Search SharePoint for relevant documents.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var userInput = GetEnv("SHAREPOINT_USER_INPUT") ?? "Find information about company policies.";

            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                userInput,
                $"SharePoint E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: UnsupportedToolEvaluators,
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "relevance",
                    "task_completion",
                }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
