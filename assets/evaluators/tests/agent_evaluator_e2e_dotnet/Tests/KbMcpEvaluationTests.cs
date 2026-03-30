// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with KB MCP (Knowledge Base) tool.
/// Uses an MCP server backed by a knowledge base.
/// Requires: MCP_KB_SERVER_URL, MCP_KB_PROJECT_CONNECTION_ID
/// </summary>
public class KbMcpEvaluationTests : AgentEvaluatorTestBase
{
    public KbMcpEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithKbMcp()
    {
        if (!HasEnv("MCP_KB_SERVER_URL", "MCP_KB_PROJECT_CONNECTION_ID"))
            throw new InvalidOperationException("Missing environment variables: MCP_KB_SERVER_URL, MCP_KB_PROJECT_CONNECTION_ID");

        var serverUrl = RequireEnv("MCP_KB_SERVER_URL");
        var connectionId = RequireEnv("MCP_KB_PROJECT_CONNECTION_ID");

        var agentName = UniqueName("E2E-KBMCP");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a helpful assistant that uses the knowledge-base retrieve tool to answer questions about earth and geology. You MUST call the tool.",
            new
            {
                type = "mcp",
                server_label = "KnowledgeBase",
                server_url = serverUrl,
                project_connection_id = connectionId,
                require_approval = new
                {
                    never = new { tool_names = new[] { "knowledge_base_retrieve" } },
                },
            }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "What's the size of Africa?",
                $"KB MCP E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "tool_output_utilization",
                    "tool_call_success",
                },
                toleratedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "groundedness",
                    "task_adherence",
                }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
