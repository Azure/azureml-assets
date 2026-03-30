// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Browser Automation tool.
/// Requires: BROWSER_AUTOMATION_PROJECT_CONNECTION_ID
/// </summary>
public class BrowserAutomationEvaluationTests : AgentEvaluatorTestBase
{
    public BrowserAutomationEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithBrowserAutomation()
    {
        if (!HasEnv("BROWSER_AUTOMATION_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variable: BROWSER_AUTOMATION_PROJECT_CONNECTION_ID");

        var connectionId = RequireEnv("BROWSER_AUTOMATION_PROJECT_CONNECTION_ID");

        var tool = new BrowserAutomationPreviewTool(
            new BrowserAutomationToolParameters(
                new BrowserAutomationToolConnectionParameters(connectionId)
            )
        );

        var agentName = UniqueName("E2E-BrowserAutomation");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant that can automate browser tasks.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "Navigate to https://example.com and tell me what the page title says.",
                $"Browser Automation E2E - {agentName}"
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
