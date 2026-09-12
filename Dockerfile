# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Set up a non-root user to run the app (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860 as required by Hugging Face Spaces
EXPOSE 7860

# Run the web service on container startup using gunicorn
# 1 worker and 8 threads is a good starting point for simple apps
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "8", "--timeout", "0", "app:app"]
