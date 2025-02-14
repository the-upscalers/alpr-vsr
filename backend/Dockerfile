# Use Python as the base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the FastAPI and Celery app code into the container
COPY . .

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libgl1 libglib2.0-0

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]