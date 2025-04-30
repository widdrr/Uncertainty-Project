# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY graphs.py .

# Create plots directory
RUN mkdir plots

# Run the script
CMD ["python", "graphs.py"]