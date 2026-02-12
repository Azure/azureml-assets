using System;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos;
using Microsoft.Extensions.Logging;

namespace Microsoft365Publisher
{
    /// <summary>
    /// Represents a request to publish data to Microsoft 365 services.
    /// </summary>
    public class Microsoft365PublishRequest
    {
        private readonly ILogger<Microsoft365PublishRequest> _logger;
        private readonly CosmosClient _cosmosClient;
        private readonly string _databaseName;
        private readonly string _containerName;

        /// <summary>
        /// Gets the FooBar string value.
        /// </summary>
        public string FooBar { get; init; }

        /// <summary>
        /// Gets the unique identifier for this request.
        /// </summary>
        public string Id { get; init; }

        /// <summary>
        /// Gets the timestamp when this request was created.
        /// </summary>
        public DateTime Timestamp { get; init; }

        /// <summary>
        /// Initializes a new instance of the Microsoft365PublishRequest class.
        /// </summary>
        /// <param name="logger">The logger instance for logging.</param>
        /// <param name="cosmosClient">The Cosmos DB client instance.</param>
        /// <param name="databaseName">The name of the Cosmos DB database.</param>
        /// <param name="containerName">The name of the Cosmos DB container.</param>
        /// <param name="fooBar">The FooBar string value to process.</param>
        public Microsoft365PublishRequest(
            ILogger<Microsoft365PublishRequest> logger,
            CosmosClient cosmosClient,
            string databaseName,
            string containerName,
            string fooBar)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _cosmosClient = cosmosClient ?? throw new ArgumentNullException(nameof(cosmosClient));
            _databaseName = databaseName ?? throw new ArgumentNullException(nameof(databaseName));
            _containerName = containerName ?? throw new ArgumentNullException(nameof(containerName));
            
            FooBar = fooBar ?? throw new ArgumentNullException(nameof(fooBar));
            Id = Guid.NewGuid().ToString();
            Timestamp = DateTime.UtcNow;

            // Log the FooBar value
            _logger.LogInformation("Microsoft365PublishRequest created with FooBar: {FooBar}", FooBar);
        }

        /// <summary>
        /// Processes the request by saving it to Cosmos DB.
        /// </summary>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ProcessAsync()
        {
            try
            {
                _logger.LogInformation("Processing Microsoft365PublishRequest with Id: {Id}, FooBar: {FooBar}", Id, FooBar);

                // Get the container
                var container = _cosmosClient.GetContainer(_databaseName, _containerName);

                // Create the document to save
                var document = new
                {
                    id = Id,
                    fooBar = FooBar,
                    timestamp = Timestamp,
                    partitionKey = Id
                };

                // Save to Cosmos DB
                var response = await container.CreateItemAsync(document, new PartitionKey(Id));

                _logger.LogInformation(
                    "Successfully saved Microsoft365PublishRequest to Cosmos DB. Id: {Id}, FooBar: {FooBar}, StatusCode: {StatusCode}",
                    Id,
                    FooBar,
                    response.StatusCode);
            }
            catch (CosmosException cosmosEx)
            {
                _logger.LogError(
                    cosmosEx,
                    "Cosmos DB error while processing Microsoft365PublishRequest. Id: {Id}, FooBar: {FooBar}, StatusCode: {StatusCode}",
                    Id,
                    FooBar,
                    cosmosEx.StatusCode);
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(
                    ex,
                    "Error while processing Microsoft365PublishRequest. Id: {Id}, FooBar: {FooBar}",
                    Id,
                    FooBar);
                throw;
            }
        }

        /// <summary>
        /// Creates and processes a Microsoft365PublishRequest.
        /// </summary>
        /// <param name="logger">The logger instance for logging.</param>
        /// <param name="cosmosClient">The Cosmos DB client instance.</param>
        /// <param name="databaseName">The name of the Cosmos DB database.</param>
        /// <param name="containerName">The name of the Cosmos DB container.</param>
        /// <param name="fooBar">The FooBar string value to process.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static async Task CreateAndProcessAsync(
            ILogger<Microsoft365PublishRequest> logger,
            CosmosClient cosmosClient,
            string databaseName,
            string containerName,
            string fooBar)
        {
            var request = new Microsoft365PublishRequest(logger, cosmosClient, databaseName, containerName, fooBar);
            await request.ProcessAsync();
        }
    }
}
