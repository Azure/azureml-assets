// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Web Search tool.
/// No external connection IDs required.
/// </summary>
public class WebSearchEvaluationTests : AgentEvaluatorTestBase
{
    public WebSearchEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithWebSearch()
    {
        var agentName = UniqueName("E2E-WebSearch");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful assistant. Use web search to answer questions with current information.",
            new
            {
                type = "web_search_preview",
                user_location = new { type = "approximate", country = "US", city = "Seattle", region = "Washington" },
            }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "What is the current population of Seattle, Washington?",
                $"Web Search E2E - {agentName}",
                toolChoice: "auto"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: UnsupportedToolEvaluators,
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "task_adherence" }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
