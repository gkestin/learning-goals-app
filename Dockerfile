FROM python:3.9-slim

# No WORKDIR, use default root

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories
RUN mkdir -p instance/uploads

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Expose port
EXPOSE 8080

# Run the application with Gunicorn using simple command line options
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 --limit-request-line 8190 --limit-request-field_size 8190 main:app 