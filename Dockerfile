# Use the explicitly requested Python version
FROM python:3.11.7-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to the terminal (useful for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for building certain Python packages like pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies first
# This leverages Docker's layer caching to speed up future builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Backend python scripts into /app
COPY Backend/ /app/

# Copy the Data CSVs directly into /app
# This ensures dataset_service.py finds 'companies.csv' in its current directory
COPY Data/ /app/

# Extract all .tar files and then delete them
RUN find /app -maxdepth 1 -name "*.tar" -exec tar -xf {} -C /app \; -exec rm -f {} \;

# --- POSTGRESQL DATA IMPORT SCRIPT ---
# Generate an entrypoint script using a heredoc to safely write the script
# This avoids the "echo \c" bug where newlines are accidentally suppressed
RUN cat <<'EOF' > /app/entrypoint.sh
#!/bin/bash
echo "Waiting for PostgreSQL to start..."
until psql -h db -U admin -d stock_data -c '\q' 2>/dev/null; do
  sleep 2
done

echo "Creating tables..."
psql -h db -U admin -d stock_data -c "CREATE TABLE IF NOT EXISTS companies (stock_code VARCHAR(50), company_name VARCHAR(255));"
psql -h db -U admin -d stock_data -c "CREATE TABLE IF NOT EXISTS stock_prices (time TIMESTAMP, symbol VARCHAR(50), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT);"

echo "Clearing old data to prevent duplicates..."
psql -h db -U admin -d stock_data -c "TRUNCATE TABLE companies, stock_prices;"

echo "Importing CSV data..."
# \copy is a psql meta-command and does not require a trailing semicolon
psql -h db -U admin -d stock_data -c "\copy companies FROM '/app/companies.csv' DELIMITER ',' CSV HEADER"
psql -h db -U admin -d stock_data -c "\copy stock_prices FROM '/app/stock_prices.csv' DELIMITER ',' CSV HEADER"

echo "Starting server..."
# exec "$@" ensures that the CMD passed by docker-compose (like Celery or Uvicorn) takes over process ID 1
exec "$@"
EOF

# Make the script executable AND strip out any Windows CRLF line endings
# that might have been inherited from the Dockerfile
RUN chmod +x /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh

# Use the script as the container's entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command for the backend container (can be overridden by celery in docker-compose)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]