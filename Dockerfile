FROM python:3.12.8-slim

# Create and switch to the app directory
WORKDIR /app

# Copy only pyproject.toml and poetry.lock first
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies (system-wide, since we turn off virtualenv creation)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Now copy your entire source code
COPY . /app

# Finally, run your server
CMD ["python", "src/server.py"]