# LangGraph Agent Deployment Demo

This repository demonstrates how to deploy a LangGraph AI agent with memory capabilities using FastAPI, PostgreSQL, and Docker. The agent uses Claude 3.5 Sonnet and can maintain conversation context through a PostgreSQL database.

## Prerequisites

- Python 3.11+
- Poetry (Python package manager)
- Docker and Docker Compose
- Anthropic API key

## Environment Setup

1. Clone this repository:

2. Create a `.env` file in the root directory with:
ANTHROPIC_API_KEY=
X_API_KEY=
DATABASE_URL=

## Local Development

### Using Poetry

1. Install dependencies:
```bash
poetry install
```
2. Run the server:
```bash
poetry run python src/server.py
```

### Using Docker Compose (Recommended)

1. Build and start the containers:
```bash
docker compose up --build
```

This will start both the FastAPI application and PostgreSQL database.

## Testing the Agent

Once running, you can test the agent using curl:

```bash
curl --location 'localhost:8080/generate' \
    --header 'X_API_KEY: <YOUR_X_API_KEY>' \
    --header 'Content-Type: application/json' \
    --data '{
    "messages":[
      {
        "type": "human",
        "content": "hello"
      }
    ]
}'
```


## Checking Agent Memory

To inspect the agent's memories in PostgreSQL:

1. Connect to the database:
```bash
psql postgresql://langgraph_demo:langgraph_demo@localhost:5432/postgres
```

## Features

- FastAPI-based API endpoints
- PostgreSQL for persistent memory storage
- Docker containerization
- API key authentication
- Claude 3.5 Sonnet integration
- Automatic memory management

## API Endpoints

- `GET /` - Redirects to API documentation
- `POST /generate` - Main endpoint for interacting with the agent
  - Requires `X_API_KEY` header
  - Accepts JSON payload with messages

## Deployment

This project is configured for deployment on fly.io. Follow these steps:

1. Install the fly.io CLI
2. Run `fly launch` in the project directory
3. Choose deployment settings and create a managed PostgreSQL database
4. Deploy with `fly deploy`

## License

MIT License
