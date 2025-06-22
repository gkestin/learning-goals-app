# Lightweight app image - builds quickly on top of the base image
FROM gcr.io/learninggoals2/learning-goals-base:latest

# Set working directory
WORKDIR /app

# Copy application code (this is the only layer that changes frequently)
COPY . .

# Create necessary directories
RUN mkdir -p instance/uploads

# Set runtime environment variables
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 120 --limit-request-line 8190 --limit-request-field_size 8190 main:app 