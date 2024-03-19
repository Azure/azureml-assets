# Need python 3.11 as the project requires it
FROM mcr.microsoft.com/devcontainers/python:3.11

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

RUN pip install uvicorn

WORKDIR /app

EXPOSE 8080

RUN ls -la

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/bin/sh", "/app/entrypoint.sh"]
