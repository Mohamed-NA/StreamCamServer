# WebSocket Secure (WSS) Camera Stream Application

[![Test and Publish Docker Image](https://github.com/MohamedEshmawy/StreamCamServer/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/MohamedEshmawy/StreamCamServer/actions/workflows/docker-publish.yml)

This project provides a secure, real-time camera streaming application using Flask and WebSockets over HTTPS (WSS).

---

## **Docker Support**

Build the container locally:

```bash
docker build -t streamcamserver:local .
```

The image already includes the certificates from the repository, so you can run it directly:

```bash
docker run --rm -p 5000:5000 streamcamserver:local
```

## **GitHub Actions Docker Publish Workflow**

This repository now includes a workflow at `.github/workflows/docker-publish.yml` that:

1. Runs the smoke test on every push to any branch.
2. Runs the smoke test and pushes a Docker image on pull requests.
3. Lets you manually trigger a test-and-push run from the **Actions** tab on any branch.

### **Required GitHub Secrets**

Add these repository secrets in GitHub under **Settings > Secrets and variables > Actions**:

- `DOCKERHUB_USERNAME`: your Docker Hub username.
- `DOCKERHUB_TOKEN`: a Docker Hub personal access token.

### **How to Create the Docker Hub Token**

1. Sign in to Docker Hub.
2. Open **Account Settings > Personal access tokens**.
3. Create a new token with write access to your repositories.
4. Copy that token and save it as the `DOCKERHUB_TOKEN` GitHub secret.

The workflow pushes images to:

```text
docker.io/<DOCKERHUB_USERNAME>/streamcamserver:latest
```

Each successful publish updates the `latest` tag.

To manually trigger the workflow:

1. Open the repository on GitHub.
2. Go to **Actions**.
3. Select **Test and Publish Docker Image**.
4. Click **Run workflow** and choose the branch you want.

If a pull request comes from a fork, GitHub does not expose repository secrets to that run, so the Docker push step will not be able to authenticate to Docker Hub.

## **Steps to Set Up the Application**

1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedEshmawy/StreamCamServer
   cd StreamCamServer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   Linux/macOS (bash/zsh):
   ```bash
   source .venv/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Windows (Command Prompt):
   ```cmd
   .venv\Scripts\activate.bat
   ```

4. Install the required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

5. Install OpenSSL on Windows (if applicable):
   - **Install via Precompiled Binaries**:
     - Download OpenSSL for Windows from [Shining Light Productions](https://slproweb.com/products/Win32OpenSSL.html).
     - Add OpenSSL to your system's `PATH` environment variable.

   - **Verify Installation**:
     - Open a terminal and run:
       ```bash
       openssl version
       ```
     - You should see the installed OpenSSL version.

6. Generate SSL certificates:
   - Open a terminal in the project directory.
   - Run the following command to create the certificates in the `certificates/` folder:
     ```bash
     mkdir certificates
     openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
         -keyout certificates/private.key \
         -out certificates/certificate.crt
     ```
   - You will be prompted to provide details such as country, organization name, etc. leave them blank for development purposes.
   - The private key will be saved as `certificates/private.key`.
   - The certificate will be saved as `certificates/certificate.crt`.

7. Start the Flask application with HTTPS and WebSocket Secure (WSS):
   ```bash
   python server.py
   ```

8. Open the application in your browser:
   - Navigate to:
     ```
     https://<your-domain-or-ip>:5000
     ```
   - The application should load, and the camera stream will be displayed.

Only port `5000` needs to be published for this app. HTTPS and WSS both use the same Flask-SocketIO listener, so there is no separate HTTP server on port `80` in the current implementation.
