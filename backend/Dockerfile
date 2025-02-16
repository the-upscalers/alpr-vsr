# Use Python as the base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libgl1 libglib2.0-0

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]