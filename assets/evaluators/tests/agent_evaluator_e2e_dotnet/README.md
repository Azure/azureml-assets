# Agent Evaluator E2E Tests (.NET)

.NET C# agent evaluator end-to-end tests.  
Each test creates an agent with a specific tool type, generates a response,
then runs all 12 quality evaluators against the response.

## Prerequisites

- .NET 8.0 SDK
- Azure AI Project with deployed models
- Azure credentials configured for `DefaultAzureCredential`

## Setup

1. Copy `.env.example` to `.env` and fill in your values:
   ```
   copy .env.example .env
   ```

2. The `assets/` folder is shared with the sibling project and contains:
   - `sample_product_info.txt` – product catalog for File Search test
   - `weather_openapi.json` – OpenAPI spec for the weather API test
   - `cua_*.png` – screenshots for the Computer Use test

## Running Tests

```bash
dotnet test
```

Run a specific test:
```bash
dotnet test --filter "FullyQualifiedName~FunctionToolEvaluationTests"
```

## Test Matrix

| Test Class | Tool Type | Required Env Vars |
|---|---|---|
| `FunctionToolEvaluationTests` | FunctionTool | _(none)_ |
| `CodeInterpreterEvaluationTests` | CodeInterpreter | _(none)_ |
| `FileSearchEvaluationTests` | FileSearch | _(none)_ |
| `WebSearchEvaluationTests` | WebSearch | _(none)_ |
| `AiSearchEvaluationTests` | Azure AI Search | `AI_SEARCH_PROJECT_CONNECTION_ID`, `AI_SEARCH_INDEX_NAME` |
| `BingGroundingEvaluationTests` | Bing Grounding | `BING_PROJECT_CONNECTION_ID` |
| `BingCustomSearchEvaluationTests` | Bing Custom Search | `BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID`, `BING_CUSTOM_SEARCH_INSTANCE_NAME` |
| `SharepointEvaluationTests` | SharePoint | `SHAREPOINT_PROJECT_CONNECTION_ID` |
| `FabricEvaluationTests` | Microsoft Fabric | `FABRIC_PROJECT_CONNECTION_ID` |
| `BrowserAutomationEvaluationTests` | Browser Automation | `BROWSER_AUTOMATION_PROJECT_CONNECTION_ID` |
| `ComputerUseEvaluationTests` | Computer Use | `COMPUTER_USE_MODEL_DEPLOYMENT_NAME` |
| `ImageGenerationEvaluationTests` | Image Generation | `IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME` |
| `MemorySearchEvaluationTests` | Memory Search | `MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME`, `MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME` |
| `McpEvaluationTests` | MCP (Microsoft Learn) | _(defaults available)_ |
| `KbMcpEvaluationTests` | MCP (Knowledge Base) | `MCP_KB_SERVER_URL`, `MCP_KB_PROJECT_CONNECTION_ID` |
| `OpenApiEvaluationTests` | OpenAPI | _(none)_ |
| `AgentToAgentEvaluationTests` | Agent-to-Agent | `A2A_PROJECT_CONNECTION_ID` |
| `AzureFunctionEvaluationTests` | Azure Function | `STORAGE_INPUT_QUEUE_NAME`, `STORAGE_OUTPUT_QUEUE_NAME`, `STORAGE_QUEUE_SERVICE_ENDPOINT` |

## Architecture

- **`AgentEvaluatorTestBase.cs`** – Shared base class.
  Contains client initialization, evaluation runner, result assertions, and constants.
- **`Tests/*.cs`** – One test class per tool type, each with a single `[Fact]` test method.

## Packages

| Package | Version |
|---|---|
| `Azure.AI.Projects` | 2.0.0 |
| `Azure.AI.Projects.Agents` | 2.0.0 |
| `Azure.Identity` | 1.13.2 |
| `xunit` | 2.9.3 |
| `DotNetEnv` | 3.1.1 |
