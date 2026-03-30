// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with a Function Tool.
/// The agent is created with a locally-defined function tool (get_horoscope).
/// The evaluation system invokes the agent directly via azure_ai_target_completions.
/// </summary>
public class FunctionToolEvaluationTests : AgentEvaluatorTestBase
{
    public FunctionToolEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithFunctionTool()
    {
        var agentName = UniqueName("E2E-FunctionTool");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful assistant that can use function tools.",
            new
            {
                type = "function",
                name = "get_horoscope",
                description = "Get today's horoscope for an astrological sign.",
                parameters = new
                {
                    type = "object",
                    properties = new
                    {
                        sign = new
                        {
                            type = "string",
                            description = "An astrological sign like Taurus or Aquarius",
                        },
                    },
                    required = new[] { "sign" },
                    additionalProperties = false,
                },
                strict = true,
            }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "What is my horoscope? I am an Aquarius.",
                $"Function Tool E2E - {agentName}"
            );
            AssertEvaluationResults(outputItems);
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
