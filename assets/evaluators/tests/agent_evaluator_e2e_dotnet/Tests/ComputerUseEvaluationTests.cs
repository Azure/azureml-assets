// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Computer Use tool.
/// The evaluation system drives the agent directly via azure_ai_target_completions.
/// Requires: COMPUTER_USE_MODEL_DEPLOYMENT_NAME (defaults to 'computer-use-preview')
/// </summary>
public class ComputerUseEvaluationTests : AgentEvaluatorTestBase
{
    public ComputerUseEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithComputerUse()
    {
        if (!HasEnv("COMPUTER_USE_MODEL_DEPLOYMENT_NAME"))
            throw new InvalidOperationException("Missing environment variable: COMPUTER_USE_MODEL_DEPLOYMENT_NAME");

        var computerModel = GetEnv("COMPUTER_USE_MODEL_DEPLOYMENT_NAME") ?? "computer-use-preview";

        var agentName = UniqueName("E2E-ComputerUse");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, computerModel,
            "You are a computer automation assistant. " +
            "Be direct and efficient. When you reach the search results page, " +
            "read and describe the actual search result titles and descriptions you can see.",
            new { type = "computer_use_preview", display_width = 1026, display_height = 769, environment = "windows" }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "I need you to help me search for 'OpenAI news'. " +
                "Please type 'OpenAI news' and submit the search. " +
                "Once you see search results, the task is complete.",
                $"Computer Use E2E - {agentName}",
                toolChoice: "auto",
                truncation: "auto"
            );
            AssertEvaluationResults(
                outputItems,
                expectedErrors: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["coherence"] = "arguments' field must be a dictionary",
                    ["fluency"] = "arguments' field must be a dictionary",
                    ["groundedness"] = "arguments' field must be a dictionary",
                    ["relevance"] = "arguments' field must be a dictionary",
                    ["intent_resolution"] = "arguments' field must be a dictionary",
                    ["task_adherence"] = "arguments' field must be a dictionary",
                    ["task_completion"] = "arguments' field must be a dictionary",
                    ["tool_call_success"] = "arguments' field must be a dictionary",
                    ["tool_call_accuracy"] = "Tool definitions input is required but not provided",
                    ["tool_selection"] = "Tool definitions input is required but not provided",
                    ["tool_input_accuracy"] = "arguments' field must be a dictionary",
                    ["tool_output_utilization"] = "arguments' field must be a dictionary",
                }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
