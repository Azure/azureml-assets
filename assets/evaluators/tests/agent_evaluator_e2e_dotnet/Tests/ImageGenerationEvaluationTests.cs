// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Xunit;
using Xunit.Abstractions;

namespace AgentEvaluatorE2E;

/// <summary>
/// E2E evaluation test for an agent with Image Generation tool.
/// Requires: IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME
/// </summary>
public class ImageGenerationEvaluationTests : AgentEvaluatorTestBase
{
    public ImageGenerationEvaluationTests(ITestOutputHelper output) : base(output) { }

    [RetryFact(MaxRetries = 3)]
    public void EvaluateAgentWithImageGeneration()
    {
        if (!HasEnv("IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME"))
            throw new InvalidOperationException("Missing environment variable: IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME");

        var imageModel = RequireEnv("IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME");

        var agentName = UniqueName("E2E-ImageGen");

        var (name, version) = CreateAgentVersionProtocol(
            agentName, ModelDeploymentName,
            "You are a creative assistant that generates images based on descriptions.",
            new { type = "image_generation", model = imageModel, quality = "low", size = "1024x1024" }
        );

        try
        {
            var (evalId, runId, outputItems) = RunEvaluation(
                name, version,
                "Generate an image of a sunset over a mountain lake.",
                $"Image Generation E2E - {agentName}",
                toolChoice: "auto",
                extraHeaders: new Dictionary<string, string>
                {
                    ["x-ms-oai-image-generation-deployment"] = imageModel,
                }
            );
            AssertEvaluationResults(
                outputItems,
                expectedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "intent_resolution",
                    "tool_input_accuracy",
                },
                toleratedFailures: new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "task_completion",
                }
            );
        }
        finally
        {
            ProjectClient.Agents.DeleteAgentVersion(name, version);
        }
    }
}
