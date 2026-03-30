// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Agent-to-Agent (A2A) tool.
/// Requires: A2A_PROJECT_CONNECTION_ID
/// </summary>
public class AgentToAgentEvaluationTests : AgentEvaluatorTestBase
{
    public AgentToAgentEvaluationTests(ITestOutputHelper output) : base(output) { }

    [Fact]
    public void EvaluateAgentWithA2A()
    {
        if (!HasEnv("A2A_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variable: A2A_PROJECT_CONNECTION_ID");

        var connectionId = RequireEnv("A2A_PROJECT_CONNECTION_ID");

        var tool = new A2APreviewTool { ProjectConnectionId = connectionId };

        var a2aEndpoint = GetEnv("A2A_ENDPOINT");
        if (!string.IsNullOrEmpty(a2aEndpoint))
            tool.BaseUri = new Uri(a2aEndpoint);

        var agentName = UniqueName("E2E-A2A");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a coordinator agent. Delegate tasks to connected agents when appropriate.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var userInput = GetEnv("A2A_USER_INPUT") ?? "What capabilities do you have through your connected agents?";

            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                userInput,
                $"Agent-to-Agent E2E - {agentName}"
            );
            AssertEvaluationResults(outputItems);
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
