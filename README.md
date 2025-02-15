# Automatic License Plate Recognition with Video Super Resolution

## Setting up Backend (With Docker Compose)

#### 1. Change directory

```sh
cd backend
```

#### 2. Build Docker image

```sh
docker-compose build
```

#### 3. Run Docker image

```sh
docker-compose up
```

`Note:` If you want to make changes to the backend code, you can restart the backend service by running the following command:

```sh
docker-compose up --force-recreate
```

If you install any new dependencies, you need to rebuild the Docker image by running the following command:

```sh
docker-compose up --build
```

## Setting up Backend (Without Docker Compose)

#### 1. Change directory

```sh
cd backend
```

#### 2. Create virtual environment

```sh
 python -m venv .venv
```

#### 3. Activate virtual environment

```sh
# For Linux/MacOS
source .venv/bin/activate

# For Windows
.venv/Scripts/activate.bat # In CMD
.venv/Scripts/Activate.ps1 # In Powershell
```

#### 4. Install dependencies

```sh
pip install -r requirements.txt
```

#### 5. Create RabbitMQ container

```sh
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

#### 6. Start Celery Worker

```sh
celery -A tasks worker --loglevel=info -P threads
```

#### 7. Start FastAPI

```sh
fastapi dev server.py
```

## Setting up Frontend

#### 1. Change directory

```sh
cd frontend
```

#### 2. Create virtual environment

```sh
 python -m venv .venv
```

#### 3. Activate virtual environment

```sh
# For Linux/MacOS
source .venv/bin/activate

# For Windows
.venv/Scripts/activate.bat # In CMD
.venv/Scripts/Activate.ps1 # In Powershell
```

#### 4. Install dependencies

```sh
pip install -r requirements.txt
```

#### 5. Start app

```sh
python client.py
```
