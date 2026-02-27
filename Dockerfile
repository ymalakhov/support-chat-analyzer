FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py .
COPY generate.py .
COPY analyze.py .
COPY prompts/ prompts/

# Create output directory
RUN mkdir -p output

# Default command: show help
CMD ["python", "generate.py", "--help"]
