# Use the full Python image which includes most build tools and common libs
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app/src"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["python", "src/noise_analysis.py"]
