// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with MCP (Model Context Protocol) tool.
/// Uses the Microsoft Learn MCP server.
/// </summary>
public class McpEvaluationTests : AgentEvaluatorTestBase
{
    public McpEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithMcp()
    {
        var serverUrl = GetEnv("MCP_LEARN_SERVER_URL") ?? "https://learn.microsoft.com/api/mcp";
        var connectionId = GetEnv("MCP_LEARN_PROJECT_CONNECTION_ID") ?? "MicrosoftLearn2";

        var agentName = UniqueName("E2E-MCP");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful assistant that can search Microsoft Learn documentation. Use the MCP tool to look up Azure documentation. When you find relevant pages, fetch the full content for completeness.",
            new
            {
                type = "mcp",
                server_label = "MicrosoftLearn2",
                server_url = serverUrl,
                project_connection_id = connectionId,
                require_approval = new
                {
                    never = new { tool_names = new[] { "microsoft_docs_search", "microsoft_docs_fetch" } },
                },
            }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "can you tell me more about how azure functions work? use MCP",
                $"MCP E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
