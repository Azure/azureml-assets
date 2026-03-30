// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Code Interpreter tool.
/// No external connections required.
/// </summary>
public class CodeInterpreterEvaluationTests : AgentEvaluatorTestBase
{
    public CodeInterpreterEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithCodeInterpreter()
    {
        var agentName = UniqueName("E2E-CodeInterpreter");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful math assistant. Use code interpreter to solve problems.",
            new { type = "code_interpreter" }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "Calculate the first 10 prime numbers and return them as a list.",
                $"Code Interpreter E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: UnsupportedToolEvaluators
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
