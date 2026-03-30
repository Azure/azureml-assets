// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Microsoft Fabric tool.
/// Requires: FABRIC_PROJECT_CONNECTION_ID
/// </summary>
public class FabricEvaluationTests : AgentEvaluatorTestBase
{
    public FabricEvaluationTests(ITestOutputHelper output) : base(output) { }

    [Fact]
    public void EvaluateAgentWithFabric()
    {
        if (!HasEnv("FABRIC_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variable: FABRIC_PROJECT_CONNECTION_ID");

        var connectionId = RequireEnv("FABRIC_PROJECT_CONNECTION_ID");

        var tool = new MicrosoftFabricPreviewTool(
            new FabricDataAgentToolOptions
            {
                ProjectConnections = { new ToolProjectConnection(connectionId) },
            }
        );

        var agentName = UniqueName("E2E-Fabric");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a data analyst assistant. Use Fabric to query and analyze data.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var userInput = GetEnv("FABRIC_USER_INPUT") ?? "What data is available in the connected Fabric workspace?";

            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                userInput,
                $"Fabric E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: UnsupportedToolEvaluators,
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "relevance",
                    "intent_resolution",
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
