// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.ClientModel;
using System.Net.Http.Headers;
using System.Text.Json;
using Azure.AI.Projects;
using Azure.Core;
using Azure.Identity;
using DotNetEnv;
using OpenAI.Evals;
using Xunit;
using Xunit.Abstractions;

[assembly: CollectionBehavior(MaxParallelThreads = 0)]

namespace AgentEvaluatorE2E;

/// <summary>
/// Shared test infrastructure.
/// Provides client initialization, evaluation helpers, and result assertions.
/// Uses the azure_ai_responses evaluation pattern:
///   1. Create agent version (typed or protocol-level)
///   2. Create a response by invoking the agent via the responses API
///   3. Evaluate the stored response using azure_ai_responses data source
/// </summary>
public abstract class AgentEvaluatorTestBase : IAsyncLifetime
{
    protected readonly ITestOutputHelper Output;

    // Clients initialized in InitializeAsync
    protected AIProjectClient ProjectClient { get; private set; } = null!;
    protected EvaluationClient EvaluationClient { get; private set; } = null!;
    private DefaultAzureCredential _credential = null!;

    protected string ModelDeploymentName { get; private set; } = null!;
    protected string ProjectEndpoint { get; private set; } = null!;

    // All 12 evaluators
    protected static readonly string[] EvaluatorNames =
    [
        "coherence",
        "fluency",
        "groundedness",
        "relevance",
        "intent_resolution",
        "task_adherence",
        "task_completion",
        "tool_call_success",
        "tool_call_accuracy",
        "tool_selection",
        "tool_input_accuracy",
        "tool_output_utilization",
    ];

    /// <summary>
    /// Evaluators that set check_for_unsupported_tools = True and return
    /// NOT_APPLICABLE for tool types in ConversationValidator.UNSUPPORTED_TOOLS.
    /// </summary>
    protected static readonly HashSet<string> UnsupportedToolEvaluators = new(StringComparer.OrdinalIgnoreCase)
    {
        "tool_call_accuracy",
        "tool_input_accuracy",
        "tool_output_utilization",
        "tool_call_success",
        "groundedness",
    };

    private const int EvalPollIntervalMs = 5_000;
    private const int EvalTimeoutMs = 600_000; // 10 minutes

    protected AgentEvaluatorTestBase(ITestOutputHelper output)
    {
        Output = output;
    }

    public Task InitializeAsync()
    {
        // Load .env from the project root (copied to output dir)
        var envPath = Path.Combine(AppContext.BaseDirectory, ".env");
        if (File.Exists(envPath))
            Env.Load(envPath);

        ProjectEndpoint = RequireEnv("AZURE_AI_PROJECT_ENDPOINT");
        ModelDeploymentName = RequireEnv("AZURE_AI_MODEL_DEPLOYMENT_NAME");

        _credential = new DefaultAzureCredential();
        ProjectClient = new AIProjectClient(new Uri(ProjectEndpoint), _credential);
        EvaluationClient = ProjectClient.OpenAI.GetEvaluationClient();

        return Task.CompletedTask;
    }

    public Task DisposeAsync() => Task.CompletedTask;

    // ------------------------------------------------------------------
    // Environment helpers
    // ------------------------------------------------------------------

    protected static string RequireEnv(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(value))
            throw new InvalidOperationException($"Missing required environment variable: {name}");
        return value;
    }

    protected static string? GetEnv(string name) =>
        Environment.GetEnvironmentVariable(name);

    protected static bool HasEnv(params string[] names) =>
        names.All(n => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable(n)));

    protected static string UniqueName(string prefix) =>
        $"{prefix}-{Guid.NewGuid():N}"[..Math.Min(prefix.Length + 9, 64)];

    // ------------------------------------------------------------------
    // Agent creation helpers
    // ------------------------------------------------------------------

    /// <summary>
    /// Creates an agent version using the protocol-level API. Use this for tools
    /// whose typed wrappers (e.g. InternalCodeInterpreterTool) are internal.
    /// Returns (agentName, agentVersion) strings for use with RunEvaluation and cleanup.
    /// </summary>
    protected (string Name, string Version) CreateAgentVersionProtocol(
        string agentName, string model, string instructions, params object[] tools)
    {
        var body = BinaryData.FromObjectAsJson(new
        {
            definition = new { kind = "prompt", model, instructions, tools },
        });
        using var content = BinaryContent.Create(body);
        var result = ProjectClient.Agents.CreateAgentVersion(agentName, content, null, new());
        var fields = ParseFields(result, "name", "version");
        return (fields["name"], fields["version"]);
    }

    // ------------------------------------------------------------------
    // Conversation helpers (for multi-turn tests like Memory Search)
    // ------------------------------------------------------------------

    /// <summary>
    /// Creates a conversation with a single user message via the conversations API.
    /// Returns the conversation ID.
    /// </summary>
    protected string CreateConversation(string userMessage)
    {
        var tokenContext = new TokenRequestContext(["https://ai.azure.com/.default"]);
        var token = _credential.GetToken(tokenContext, default);

        var body = JsonSerializer.Serialize(new
        {
            items = new[]
            {
                new { type = "message", role = "user", content = userMessage }
            }
        });

        var url = $"{ProjectEndpoint}/openai/v1/conversations";
        using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", token.Token);
        using var httpContent = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        var httpResponse = httpClient.PostAsync(url, httpContent).GetAwaiter().GetResult();

        var responseBody = httpResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        if (!httpResponse.IsSuccessStatusCode)
        {
            Output.WriteLine($"Conversation creation failed ({httpResponse.StatusCode}): {responseBody}");
            httpResponse.EnsureSuccessStatusCode();
        }

        var doc = JsonDocument.Parse(responseBody);
        var convId = doc.RootElement.GetProperty("id").GetString()!;
        Output.WriteLine($"Conversation created: {convId}");
        return convId;
    }

    /// <summary>
    /// Invokes the agent via the responses API within an existing conversation.
    /// Returns the response ID.
    /// </summary>
    protected string CreateResponseForAgentWithConversation(string agentName, string conversationId)
    {
        var tokenContext = new TokenRequestContext(["https://ai.azure.com/.default"]);
        var token = _credential.GetToken(tokenContext, default);

        var body = JsonSerializer.Serialize(new
        {
            conversation = conversationId,
            agent_reference = new { name = agentName, type = "agent_reference" },
        });

        var url = $"{ProjectEndpoint}/openai/v1/responses";
        using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", token.Token);
        using var httpContent = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        var httpResponse = httpClient.PostAsync(url, httpContent).GetAwaiter().GetResult();

        var responseBody = httpResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        if (!httpResponse.IsSuccessStatusCode)
        {
            Output.WriteLine($"Response creation failed ({httpResponse.StatusCode}): {responseBody}");
            httpResponse.EnsureSuccessStatusCode();
        }

        var doc = JsonDocument.Parse(responseBody);
        var responseId = doc.RootElement.GetProperty("id").GetString()!;
        Output.WriteLine($"Agent response (conversation={conversationId}): {responseId}");
        return responseId;
    }

    /// <summary>
    /// Runs evaluation against an already-created response ID (skips response creation).
    /// </summary>
    protected (string EvalId, string RunId, List<JsonElement> OutputItems) RunEvaluationWithResponseId(
        string responseId, string agentName, string evalName)
    {
        var testingCriteria = EvaluatorNames.Select(name => new
        {
            type = "azure_ai_evaluator",
            name,
            evaluator_name = $"builtin.{name}",
            initialization_parameters = new { deployment_name = ModelDeploymentName },
        }).ToArray();

        var evalConfig = BinaryData.FromObjectAsJson(new
        {
            name = evalName,
            data_source_config = new { type = "azure_ai_source", scenario = "responses" },
            testing_criteria = testingCriteria,
        });

        using var evalContent = BinaryContent.Create(evalConfig);
        var evalResult = EvaluationClient.CreateEvaluation(evalContent);
        var evalFields = ParseFields(evalResult, "id");
        var evaluationId = evalFields["id"];

        try
        {
            var runData = BinaryData.FromObjectAsJson(new
            {
                eval_id = evaluationId,
                name = $"E2E Run - {agentName}",
                data_source = new
                {
                    type = "azure_ai_responses",
                    item_generation_params = new
                    {
                        type = "response_retrieval",
                        data_mapping = new { response_id = "{{item.resp_id}}" },
                        source = new
                        {
                            type = "file_content",
                            content = new[] { new { item = new { resp_id = responseId } } },
                        },
                    },
                },
            });

            using var runContent = BinaryContent.Create(runData);
            var runResult = EvaluationClient.CreateEvaluationRun(evaluationId, runContent);
            var runFields = ParseFields(runResult, "id", "status");
            var runId = runFields["id"];
            var status = runFields["status"];

            Output.WriteLine($"Eval run created: {runId} (eval: {evaluationId})");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (status != "completed" && status != "failed")
            {
                if (sw.ElapsedMilliseconds > EvalTimeoutMs)
                    throw new TimeoutException($"Eval run did not complete within {EvalTimeoutMs / 1000}s. Last status: {status}");

                Thread.Sleep(EvalPollIntervalMs);
                runResult = EvaluationClient.GetEvaluationRun(evaluationId, runId, new());
                status = ParseFields(runResult, "status")["status"];
                Output.WriteLine($"Eval run status: {status} ({sw.Elapsed.TotalSeconds:F0}s elapsed)");
            }

            var runDoc = JsonDocument.Parse(runResult.GetRawResponse().Content.ToMemory());
            var runRoot = runDoc.RootElement;

            if (status == "failed")
            {
                var err = GetErrorMessage(runResult);
                var resultCounts = runRoot.TryGetProperty("result_counts", out var rc) ? PrettyJson(rc) : "N/A";
                Output.WriteLine($"Eval run FAILED. error={err}, result_counts: {resultCounts}");
            }

            Assert.True(status == "completed",
                $"Eval run did not complete. Status: {status}");

            var outputItems = GetOutputItems(evaluationId, runId);
            var completedResultCounts = runRoot.TryGetProperty("result_counts", out var crc) ? PrettyJson(crc) : "N/A";
            Output.WriteLine($"Eval completed – result_counts: {completedResultCounts}, output_items: {outputItems.Count}");

            return (evaluationId, runId, outputItems);
        }
        finally
        {
            EvaluationClient.DeleteEvaluation(evaluationId, new());
        }
    }

    /// <summary>
    /// Deletes a conversation. Swallows errors for cleanup.
    /// </summary>
    protected void DeleteConversation(string? conversationId)
    {
        if (conversationId == null) return;
        try
        {
            var tokenContext = new TokenRequestContext(["https://ai.azure.com/.default"]);
            var token = _credential.GetToken(tokenContext, default);
            var url = $"{ProjectEndpoint}/openai/v1/conversations/{conversationId}";
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
            httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token.Token);
            httpClient.DeleteAsync(url).GetAwaiter().GetResult();
            Output.WriteLine($"Conversation deleted: {conversationId}");
        }
        catch (Exception ex)
        {
            Output.WriteLine($"Failed to delete conversation {conversationId}: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Response creation: invoke agent via responses API
    // ------------------------------------------------------------------

    /// <summary>
    /// Invokes the agent via the OpenAI-compatible responses API and returns
    /// the stored response ID.
    /// </summary>
    protected string CreateResponseForAgent(
        string agentName, string query,
        string toolChoice = "required",
        Dictionary<string, string>? extraHeaders = null,
        string? truncation = null)
    {
        var tokenContext = new TokenRequestContext(["https://ai.azure.com/.default"]);
        var token = _credential.GetToken(tokenContext, default);

        var bodyDict = new Dictionary<string, object>
        {
            ["input"] = query,
            ["tool_choice"] = toolChoice,
            ["agent_reference"] = new { name = agentName, type = "agent_reference" },
        };
        if (truncation != null)
            bodyDict["truncation"] = truncation;

        var requestBody = JsonSerializer.Serialize(bodyDict);
        var url = $"{ProjectEndpoint}/openai/v1/responses";

        // Retry with exponential backoff on 429 (rate limiting)
        const int maxRetries = 3;
        for (int attempt = 0; ; attempt++)
        {
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
            httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token.Token);

            if (extraHeaders != null)
            {
                foreach (var (key, value) in extraHeaders)
                    httpClient.DefaultRequestHeaders.TryAddWithoutValidation(key, value);
            }

            using var httpContent = new StringContent(requestBody, System.Text.Encoding.UTF8, "application/json");
            var httpResponse = httpClient.PostAsync(url, httpContent).GetAwaiter().GetResult();

            if ((int)httpResponse.StatusCode == 429 && attempt < maxRetries)
            {
                var retryAfter = httpResponse.Headers.RetryAfter?.Delta?.TotalMilliseconds
                    ?? (Math.Pow(2, attempt + 1) * 5_000);
                Output.WriteLine($"Rate limited (429), retrying in {retryAfter / 1000:F0}s (attempt {attempt + 1}/{maxRetries})");
                Thread.Sleep((int)retryAfter);
                continue;
            }

            var responseBody = httpResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            if (!httpResponse.IsSuccessStatusCode)
            {
                Output.WriteLine($"Response creation failed ({httpResponse.StatusCode}): {responseBody}");
                httpResponse.EnsureSuccessStatusCode();
            }

            var doc = JsonDocument.Parse(responseBody);
            var responseId = doc.RootElement.GetProperty("id").GetString()!;

            Output.WriteLine($"Agent response: {responseId}");
            Output.WriteLine(PrettyJson(doc.RootElement));

            return responseId;
        }

        // Unreachable — loop always returns or throws
        throw new InvalidOperationException("Exhausted retries without returning");
    }

    private static string PrettyJson(JsonElement element)
    {
        return JsonSerializer.Serialize(element, new JsonSerializerOptions { WriteIndented = true });
    }

    // ------------------------------------------------------------------
    // Evaluation: create eval, run against agent, poll, collect results
    // ------------------------------------------------------------------

    /// <summary>
    /// Runs all 12 evaluators against an agent using the azure_ai_responses
    /// pattern. First creates a response by
    /// invoking the agent via the responses API, then evaluates that stored
    /// response.
    /// </summary>
    protected (string EvalId, string RunId, List<JsonElement> OutputItems) RunEvaluation(
        string agentName, string agentVersion, string query, string evalName,
        string toolChoice = "required",
        Dictionary<string, string>? extraHeaders = null,
        string? truncation = null)
    {
        // Step 1: Create a response by invoking the agent
        var responseId = CreateResponseForAgent(agentName, query, toolChoice, extraHeaders, truncation);

        // Step 2: Build testing criteria for all 12 evaluators
        var testingCriteria = EvaluatorNames.Select(name => new
        {
            type = "azure_ai_evaluator",
            name,
            evaluator_name = $"builtin.{name}",
            initialization_parameters = new { deployment_name = ModelDeploymentName },
        }).ToArray();

        var evalConfig = BinaryData.FromObjectAsJson(new
        {
            name = evalName,
            data_source_config = new { type = "azure_ai_source", scenario = "responses" },
            testing_criteria = testingCriteria,
        });

        using var evalContent = BinaryContent.Create(evalConfig);
        var evalResult = EvaluationClient.CreateEvaluation(evalContent);
        var evalFields = ParseFields(evalResult, "id");
        var evaluationId = evalFields["id"];

        try
        {
            // Step 3: Run evaluation using azure_ai_responses
            var runData = BinaryData.FromObjectAsJson(new
            {
                eval_id = evaluationId,
                name = $"E2E Run - {agentName}",
                data_source = new
                {
                    type = "azure_ai_responses",
                    item_generation_params = new
                    {
                        type = "response_retrieval",
                        data_mapping = new { response_id = "{{item.resp_id}}" },
                        source = new
                        {
                            type = "file_content",
                            content = new[] { new { item = new { resp_id = responseId } } },
                        },
                    },
                },
            });

            using var runContent = BinaryContent.Create(runData);
            var runResult = EvaluationClient.CreateEvaluationRun(evaluationId, runContent);
            var runFields = ParseFields(runResult, "id", "status");
            var runId = runFields["id"];
            var status = runFields["status"];

            Output.WriteLine($"Eval run created: {runId} (eval: {evaluationId})");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (status != "completed" && status != "failed")
            {
                if (sw.ElapsedMilliseconds > EvalTimeoutMs)
                    throw new TimeoutException($"Eval run did not complete within {EvalTimeoutMs / 1000}s. Last status: {status}");

                Thread.Sleep(EvalPollIntervalMs);
                runResult = EvaluationClient.GetEvaluationRun(evaluationId, runId, new());
                status = ParseFields(runResult, "status")["status"];
                Output.WriteLine($"Eval run status: {status} ({sw.Elapsed.TotalSeconds:F0}s elapsed)");
            }

            // Parse full run result for detailed logging
            var runDoc = JsonDocument.Parse(runResult.GetRawResponse().Content.ToMemory());
            var runRoot = runDoc.RootElement;

            if (status == "failed")
            {
                var err = GetErrorMessage(runResult);
                var resultCounts = runRoot.TryGetProperty("result_counts", out var rc) ? PrettyJson(rc) : "N/A";
                var perModelUsage = runRoot.TryGetProperty("per_model_usage", out var mu) ? PrettyJson(mu) : "N/A";
                var perCriteriaResults = runRoot.TryGetProperty("per_testing_criteria_results", out var cr) ? PrettyJson(cr) : "N/A";
                Output.WriteLine($"Eval run FAILED. error={err}");
                Output.WriteLine($"  result_counts: {resultCounts}");
                Output.WriteLine($"  per_model_usage: {perModelUsage}");
                Output.WriteLine($"  per_testing_criteria_results: {perCriteriaResults}");
            }

            Assert.True(status == "completed",
                $"Eval run did not complete. Status: {status}");

            // Collect output items
            var outputItems = GetOutputItems(evaluationId, runId);
            var completedResultCounts = runRoot.TryGetProperty("result_counts", out var crc) ? PrettyJson(crc) : "N/A";
            Output.WriteLine($"Eval completed – result_counts: {completedResultCounts}, output_items: {outputItems.Count}");

            return (evaluationId, runId, outputItems);
        }
        finally
        {
            EvaluationClient.DeleteEvaluation(evaluationId, new());
        }
    }

    // ------------------------------------------------------------------
    // Result assertion
    // ------------------------------------------------------------------

    protected void AssertEvaluationResults(
        List<JsonElement> outputItems,
        HashSet<string>? expectedNotApplicable = null,
        Dictionary<string, string>? expectedErrors = null,
        HashSet<string>? expectedFailures = null,
        HashSet<string>? toleratedFailures = null)
    {
        expectedNotApplicable ??= new();
        expectedErrors ??= new(StringComparer.OrdinalIgnoreCase);
        expectedFailures ??= new(StringComparer.OrdinalIgnoreCase);
        toleratedFailures ??= new(StringComparer.OrdinalIgnoreCase);

        Assert.True(outputItems.Count > 0, "No output items produced by evaluation");

        foreach (var item in outputItems)
        {
            Assert.True(item.TryGetProperty("results", out var resultsArray),
                $"Output item has no 'results' property. JSON: {item}");

            var evaluatorResultsFound = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var unexpectedErrors = new List<string>();
            var unexpectedFailures = new List<string>();
            var unexpectedPasses = new List<string>();
            var expectedErrorsSeen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var expectedFailuresSeen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var result in resultsArray.EnumerateArray())
            {
                var name = result.GetPropertyOrDefault("name", "unknown");
                evaluatorResultsFound.Add(name);

                var label = result.GetPropertyOrDefault("label", null as string);
                if (label == null)
                {
                    var passed = result.GetNullableBool("passed");
                    if (passed.HasValue)
                        label = passed.Value ? "pass" : "fail";
                }

                var score = result.GetNullableDouble("score");
                var threshold = result.GetNullableDouble("threshold");
                var reason = result.GetPropertyOrDefault("reason", null as string);

                // Check for error in sample
                string? errorMsg = null;
                string? errorCode = null;
                if (result.TryGetProperty("sample", out var sample) &&
                    sample.TryGetProperty("error", out var errorElement) &&
                    errorElement.ValueKind == JsonValueKind.Object)
                {
                    errorMsg = errorElement.GetPropertyOrDefault("message", "unknown error");
                    errorCode = errorElement.GetPropertyOrDefault("code", "");
                }

                if (errorMsg != null)
                {
                    if (expectedNotApplicable.Contains(name))
                    {
                        Assert.Equal("FAILED_EXECUTION", errorCode);
                        Assert.Contains("not supported for", errorMsg, StringComparison.OrdinalIgnoreCase);
                        expectedErrorsSeen.Add(name);
                        Output.WriteLine($"Evaluator '{name}': NOT_APPLICABLE (expected) - code={errorCode}, message={errorMsg}, sample={PrettyJson(sample)}");
                    }
                    else if (expectedErrors.TryGetValue(name, out var expectedSubstring))
                    {
                        Assert.Contains(expectedSubstring, errorMsg, StringComparison.OrdinalIgnoreCase);
                        expectedErrorsSeen.Add(name);
                        Output.WriteLine($"Evaluator '{name}': expected error (documented) - {errorMsg}");
                    }
                    else
                    {
                        Output.WriteLine($"Evaluator '{name}': UNEXPECTED error - code={errorCode}, message={errorMsg}");
                        unexpectedErrors.Add($"'{name}': code={errorCode}, message={errorMsg}");
                    }
                }
                else if (label != "pass")
                {
                    if (expectedFailures.Contains(name))
                    {
                        Assert.True(score.HasValue, $"Evaluator '{name}' failed with non-numeric score");
                        if (threshold.HasValue)
                            Assert.True(score.Value <= threshold.Value,
                                $"Evaluator '{name}' score={score} > threshold={threshold} but label={label}");
                        expectedFailuresSeen.Add(name);
                        Output.WriteLine($"Evaluator '{name}': expected failure (score={score}, threshold={threshold}, reason={reason})");
                    }
                    else if (toleratedFailures.Contains(name))
                    {
                        Output.WriteLine($"Evaluator '{name}': tolerated failure (score={score}, threshold={threshold}, reason={reason})");
                    }
                    else
                    {
                        unexpectedFailures.Add($"'{name}': label={label}, score={score}, threshold={threshold}, reason={reason}");
                    }
                }
                else
                {
                    if (expectedFailures.Contains(name))
                    {
                        unexpectedPasses.Add($"'{name}': expected to fail but PASSED (score={score}, threshold={threshold})");
                    }
                    else if (toleratedFailures.Contains(name))
                    {
                        Output.WriteLine($"Evaluator '{name}': PASS (tolerated, score={score}, threshold={threshold})");
                    }
                    else
                    {
                        Output.WriteLine($"Evaluator '{name}': PASS (score={score}, threshold={threshold})");
                    }
                }
            }

            Assert.True(unexpectedErrors.Count == 0,
                $"{unexpectedErrors.Count} evaluator(s) errored unexpectedly:\n{string.Join("\n", unexpectedErrors)}");
            Assert.True(unexpectedFailures.Count == 0,
                $"{unexpectedFailures.Count} evaluator(s) did not pass:\n{string.Join("\n", unexpectedFailures)}");
            Assert.True(unexpectedPasses.Count == 0,
                $"{unexpectedPasses.Count} evaluator(s) were expected to fail but passed instead:\n{string.Join("\n", unexpectedPasses)}");

            // Verify all expected errors were actually seen
            var allExpected = new HashSet<string>(expectedNotApplicable, StringComparer.OrdinalIgnoreCase);
            foreach (var key in expectedErrors.Keys)
                allExpected.Add(key);
            var missingErrors = allExpected.Where(e => !expectedErrorsSeen.Contains(e)).ToList();
            Assert.True(missingErrors.Count == 0,
                $"{missingErrors.Count} evaluator(s) were expected to error but succeeded instead: {string.Join(", ", missingErrors)}");

            // Verify all expected failures were actually seen
            var missingFailures = expectedFailures.Where(f => !expectedFailuresSeen.Contains(f)).ToList();
            Assert.True(missingFailures.Count == 0,
                $"{missingFailures.Count} evaluator(s) were expected to fail but did not appear as failures: {string.Join(", ", missingFailures)}");

            var missing = EvaluatorNames.Where(n => !evaluatorResultsFound.Contains(n)).ToList();
            Assert.True(missing.Count == 0,
                $"Missing evaluator results for: {string.Join(", ", missing)}. Got results for: {string.Join(", ", evaluatorResultsFound)}");
        }
    }

    // ------------------------------------------------------------------
    // JSON parsing helpers (protocol-level, matching SDK sample pattern)
    // ------------------------------------------------------------------

    private static Dictionary<string, string> ParseFields(ClientResult result, params string[] keys)
    {
        var doc = JsonDocument.Parse(result.GetRawResponse().Content.ToMemory());
        var found = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            foreach (var key in keys)
            {
                if (prop.NameEquals(key) && prop.Value.ValueKind == JsonValueKind.String)
                    found[key] = prop.Value.GetString()!;
            }
        }

        var missing = keys.Where(k => !found.ContainsKey(k)).ToList();
        if (missing.Count > 0)
            throw new InvalidOperationException($"Keys not found in result: {string.Join(", ", missing)}");
        return found;
    }

    private static string GetErrorMessage(ClientResult result)
    {
        var doc = JsonDocument.Parse(result.GetRawResponse().Content.ToMemory());
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.NameEquals("error") && prop.Value.ValueKind == JsonValueKind.Object)
            {
                var msg = prop.Value.GetPropertyOrDefault("message", "");
                var code = prop.Value.GetPropertyOrDefault("code", "");
                return $"Message: {msg}, Code: {code}";
            }
        }
        return "";
    }

    private List<JsonElement> GetOutputItems(string evaluationId, string runId)
    {
        var items = new List<JsonElement>();
        bool hasMore;
        do
        {
            var result = EvaluationClient.GetEvaluationRunOutputItems(
                evaluationId, runId, limit: null, order: "asc",
                after: default, outputItemStatus: default, options: new());
            var doc = JsonDocument.Parse(result.GetRawResponse().Content.ToMemory());
            hasMore = false;
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.NameEquals("has_more"))
                    hasMore = prop.Value.GetBoolean();
                else if (prop.NameEquals("data") && prop.Value.ValueKind == JsonValueKind.Array)
                {
                    foreach (var element in prop.Value.EnumerateArray())
                        items.Add(element.Clone());
                }
            }
        } while (hasMore);

        return items;
    }
}

// ------------------------------------------------------------------
// JsonElement extension helpers
// ------------------------------------------------------------------

internal static class JsonElementExtensions
{
    public static string? GetPropertyOrDefault(this JsonElement element, string name, string? defaultValue)
    {
        if (element.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String)
            return prop.GetString();
        return defaultValue;
    }

    public static bool? GetNullableBool(this JsonElement element, string name)
    {
        if (element.TryGetProperty(name, out var prop))
        {
            if (prop.ValueKind == JsonValueKind.True) return true;
            if (prop.ValueKind == JsonValueKind.False) return false;
        }
        return null;
    }

    public static double? GetNullableDouble(this JsonElement element, string name)
    {
        if (element.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number)
            return prop.GetDouble();
        return null;
    }
}
