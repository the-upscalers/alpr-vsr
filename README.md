# Automatic License Plate Recognition with Video Super Resolution

## Setting up Backend

1. Change directory

```sh
cd backend
```

2. Create virtual environment

```sh
 python -m venv .venv
```

3. Activate virtual environment

```sh
# For Linux/MacOS
source env/bin/activate

# For Windows
env/Scripts/activate.bat # In CMD
env/Scripts/Activate.ps1 # In Powershell
```

4. Install dependencies

```sh
pip install -r requirements.txt
```

5. Create RabbitMQ container

```sh
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=guest \
  -e RABBITMQ_DEFAULT_PASS=guest \
  rabbitmq:3-management
```

6. Start Celery Worker

```sh
celery -A tasks worker --loglevel=info -P threads
```

7. Start FastAPI

```sh
fastapi dev server.py
```
