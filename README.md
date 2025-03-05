# Auto-Refresh Flask Application

A simple Flask application that can refresh itself by pulling the latest code from GitHub without requiring a container redeployment.

## Features

- Single page with a "Hello!" header and a "Refresh" button
- Clicking the "Refresh" button pulls the latest code from the configured GitHub repository
- Automatically restarts the application if Python files change
- Refreshes the page if HTML templates change

## Setup

1. Clone this repository
2. Update the `GIT_REPO_URL` environment variable in the Dockerfile or set it when running the container
3. Build and deploy the container

## Building and Running

```bash
# Build the Docker image
docker build -t auto-refresh-app .

# Run the container
docker run -p 5000:5000 -e GIT_REPO_URL="https://github.com/your-username/your-repo.git" auto-refresh-app
```

## How it Works

1. When the "Refresh" button is clicked, the application makes a POST request to `/api/pull-latest`
2. The server executes a `git pull` to fetch the latest code from the repository
3. If Python files have changed, the server restarts itself using `os.execv`
4. If only templates have changed, the page refreshes without a server restart
5. The UI displays status messages throughout the process

## Usage in Kubernetes

When deployed to Kubernetes, ensure that:

1. The container has permissions to execute `git pull`
2. The `GIT_REPO_URL` environment variable is set correctly
3. The container has appropriate resources allocated
# Mercedes-Chat
