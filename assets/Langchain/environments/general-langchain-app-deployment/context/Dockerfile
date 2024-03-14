# Use the official lightweight Python image.
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry==1.6.1

# Configure poetry: do not create a virtual environment inside the Docker container
RUN poetry config virtualenvs.create false

# Add uvicorn separately
RUN pip install uvicorn


# Set the working directory in Docker
WORKDIR /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Keep the entrypoint to set up at runtime
RUN ls -la

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/bin/sh", "/app/entrypoint.sh"]
