# Microsoft365PublishRequest

This project contains the `Microsoft365PublishRequest` class, which handles publishing requests to Microsoft 365 services with Cosmos DB integration.

## Features

- **FooBar String Parameter**: Accepts a string parameter named `FooBar`
- **Logging**: Logs the FooBar value when the request is created and processed
- **Cosmos DB Integration**: Saves the FooBar value to Cosmos DB with a unique ID and timestamp

## Usage

### Prerequisites

- .NET 6.0 or later
- Azure Cosmos DB account
- Microsoft.Azure.Cosmos NuGet package (v3.35.4 or later)
- Microsoft.Extensions.Logging.Abstractions NuGet package (v7.0.0 or later)

### Example

```csharp
using Microsoft.Azure.Cosmos;
using Microsoft.Extensions.Logging;
using Microsoft365Publisher;

// Initialize logger and Cosmos DB client
ILogger<Microsoft365PublishRequest> logger = // your logger instance
CosmosClient cosmosClient = new CosmosClient(connectionString);

// Create and process a request
string fooBar = "example value";
await Microsoft365PublishRequest.CreateAndProcessAsync(
    logger,
    cosmosClient,
    "YourDatabaseName",
    "YourContainerName",
    fooBar
);
```

### Alternatively, use the constructor directly

```csharp
var request = new Microsoft365PublishRequest(
    logger,
    cosmosClient,
    "YourDatabaseName",
    "YourContainerName",
    "example value"
);

await request.ProcessAsync();
```

## Class Structure

### Properties

- `FooBar` (string): The FooBar value to be processed and stored
- `Id` (string): Unique identifier for the request (auto-generated)
- `Timestamp` (DateTime): UTC timestamp when the request was created

### Methods

- `ProcessAsync()`: Processes the request by saving it to Cosmos DB
- `CreateAndProcessAsync()`: Static method to create and process a request in one call

## Cosmos DB Document Structure

When saved to Cosmos DB, the document has the following structure:

```json
{
  "id": "unique-guid",
  "fooBar": "the-foobar-value",
  "timestamp": "2026-02-12T00:00:00.000Z",
  "partitionKey": "unique-guid"
}
```

## Logging

The class logs the following events:

1. When the request is created: "Microsoft365PublishRequest created with FooBar: {FooBar}"
2. When processing starts: "Processing Microsoft365PublishRequest with Id: {Id}, FooBar: {FooBar}"
3. On successful save: "Successfully saved Microsoft365PublishRequest to Cosmos DB. Id: {Id}, FooBar: {FooBar}, StatusCode: {StatusCode}"
4. On error: Appropriate error messages with exception details

## Error Handling

The class includes comprehensive error handling:

- Null argument validation in the constructor
- Specific handling for Cosmos DB exceptions
- General exception handling with detailed logging
- All exceptions are logged and re-thrown to allow proper error handling at the caller level
