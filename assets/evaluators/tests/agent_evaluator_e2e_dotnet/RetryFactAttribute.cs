// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace AgentEvaluatorE2E;

/// <summary>
/// A custom [Fact] replacement that retries failed tests up to <see cref="MaxRetries"/> times.
/// Usage: [RetryFact(MaxRetries = 3)]
/// </summary>
[XunitTestCaseDiscoverer("AgentEvaluatorE2E.RetryFactDiscoverer", "AgentEvaluatorE2E")]
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class RetryFactAttribute : FactAttribute
{
    public int MaxRetries { get; set; } = 3;
}

public class RetryFactDiscoverer : IXunitTestCaseDiscoverer
{
    private readonly IMessageSink _diagnosticMessageSink;

    public RetryFactDiscoverer(IMessageSink diagnosticMessageSink)
    {
        _diagnosticMessageSink = diagnosticMessageSink;
    }

    public IEnumerable<IXunitTestCase> Discover(
        ITestFrameworkDiscoveryOptions discoveryOptions,
        ITestMethod testMethod,
        IAttributeInfo factAttribute)
    {
        var maxRetries = factAttribute.GetNamedArgument<int>(nameof(RetryFactAttribute.MaxRetries));
        if (maxRetries < 1) maxRetries = 3;

        yield return new RetryTestCase(
            _diagnosticMessageSink, discoveryOptions.MethodDisplayOrDefault(),
            discoveryOptions.MethodDisplayOptionsOrDefault(), testMethod, maxRetries);
    }
}

public class RetryTestCase : XunitTestCase
{
    private int _maxRetries;

    [Obsolete("Called by the deserializer; should only be called by deriving classes for de-serialization purposes")]
    public RetryTestCase() { }

    public RetryTestCase(
        IMessageSink diagnosticMessageSink,
        TestMethodDisplay defaultMethodDisplay,
        TestMethodDisplayOptions defaultMethodDisplayOptions,
        ITestMethod testMethod,
        int maxRetries)
        : base(diagnosticMessageSink, defaultMethodDisplay, defaultMethodDisplayOptions, testMethod)
    {
        _maxRetries = maxRetries;
    }

    public override async Task<RunSummary> RunAsync(
        IMessageSink diagnosticMessageSink,
        IMessageBus messageBus,
        object[] constructorArguments,
        ExceptionAggregator aggregator,
        CancellationTokenSource cancellationTokenSource)
    {
        var runCount = 0;
        RunSummary? summary = null;

        while (true)
        {
            runCount++;
            summary = await base.RunAsync(
                diagnosticMessageSink,
                new DelegatingMessageBus(messageBus, retrying: runCount < _maxRetries + 1),
                constructorArguments,
                new ExceptionAggregator(aggregator),
                cancellationTokenSource);

            if (summary.Failed == 0 || runCount > _maxRetries)
                break;

            diagnosticMessageSink.OnMessage(new DiagnosticMessage(
                $"[RetryFact] Test '{DisplayName}' failed (attempt {runCount}/{_maxRetries + 1}), retrying..."));
        }

        return summary;
    }

    public override void Serialize(IXunitSerializationInfo data)
    {
        base.Serialize(data);
        data.AddValue("MaxRetries", _maxRetries);
    }

    public override void Deserialize(IXunitSerializationInfo data)
    {
        base.Deserialize(data);
        _maxRetries = data.GetValue<int>("MaxRetries");
    }
}

/// <summary>
/// A message bus wrapper that swallows failure messages when we know we'll retry.
/// On the final attempt, all messages pass through normally.
/// </summary>
internal class DelegatingMessageBus : IMessageBus
{
    private readonly IMessageBus _inner;
    private readonly bool _retrying;

    public DelegatingMessageBus(IMessageBus inner, bool retrying)
    {
        _inner = inner;
        _retrying = retrying;
    }

    public bool QueueMessage(IMessageSinkMessage message)
    {
        // If we're going to retry, swallow test-failed messages so they
        // don't show up as failures in the runner.
        if (_retrying && message is ITestFailed)
            return true;

        return _inner.QueueMessage(message);
    }

    public void Dispose() { }
}
