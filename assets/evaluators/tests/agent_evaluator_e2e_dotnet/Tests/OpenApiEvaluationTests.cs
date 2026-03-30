// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with OpenAPI tool.
/// Uses the public wttr.in weather API (no auth required).
/// The OpenAPI spec is loaded from the assets folder.
/// </summary>
public class OpenApiEvaluationTests : AgentEvaluatorTestBase
{
    public OpenApiEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithOpenApi()
    {
        var assetDir = Path.Combine(AppContext.BaseDirectory, "assets");
        var specPath = Path.Combine(assetDir, "weather_openapi.json");
        var spec = BinaryData.FromString(File.ReadAllText(specPath));

        var tool = new OpenAPITool(
            new OpenAPIFunctionDefinition("get_weather", spec, new OpenAPIAnonymousAuthenticationDetails())
            {
                Description = "Retrieve weather information for a location.",
            }
        );

        var agentName = UniqueName("E2E-OpenAPI");
        var agent = ProjectClient.Agents.CreateAgentVersion(
            agentName,
            new AgentVersionCreationOptions(
                new PromptAgentDefinition(ModelDeploymentName)
                {
                    Instructions = "You are a helpful assistant that retrieves weather information using the API.",
                    Tools = { tool },
                }
            )
        );

        try
        {
            // openapi_call is in UNSUPPORTED_TOOLS -- some evaluators return NOT_APPLICABLE.
            var expectedNotApplicable = new HashSet<string>(UnsupportedToolEvaluators, StringComparer.OrdinalIgnoreCase);
            expectedNotApplicable.Remove("tool_call_accuracy");

            var (evalId, runId, outputItems) = RunEvaluation(
                agent.Value.Name, agent.Value.Version,
                "What's the weather like in Cairo?",
                $"OpenAPI E2E - {agentName}"
            );
            AssertEvaluationResults(
                outputItems,
                expectedNotApplicable: expectedNotApplicable,
                expectedErrors: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["tool_call_accuracy"] = "Tool definitions input is required but not provided",
                    ["tool_selection"] = "Tool definitions input is required but not provided",
                },
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "task_adherence" }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(agent.Value.Name, agent.Value.Version);
        }
    }
}
