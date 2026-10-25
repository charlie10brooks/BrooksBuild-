# Run TerraScope with Docker: step by step

**Status: not verified.** The Docker image has never been built (the environment that wrote it could not reach Docker Hub). The steps are right as far as I can tell,
but expect to fix small things on the first build. If something fails, copy the error text and send it back.

## 1. Install Docker
* **Windows (recommended route):** install Docker Desktop, turn on "Use the WSL 2 based engine", and install Ubuntu from the Microsoft Store.
  Do ALL the following steps inside the Ubuntu (WSL) terminal. Windows-style paths such as `C:\...` do not work with this setup.
* **Mac:** install Docker Desktop. Open it and wait until it says it is running.
* **Linux:** install Docker Engine and the Compose plugin (docs.docker.com/engine/install).

Check it works: `docker --version` and `docker compose version` should both print a version.

## 2. Unzip and open a terminal in the folder
```
unzip terrascope.zip
cd terrascope
```

## 3. Make a data folder and a settings file
Pick a folder where all projects, photos and results will be stored, and create it. It needs plenty of free disk (photos are large).
```
mkdir -p ~/terrascope-data
pwd            # not needed, just look at the next line
echo ~/terrascope-data   # this prints the ABSOLUTE path you need below, e.g. /home/you/terrascope-data
cp .env.example .env
```
Open `.env` in any text editor and set the line
```
TERRASCOPE_DATA_DIR=/home/you/terrascope-data
```
to the absolute path printed above (no `~`). On a Mac it looks like `/Users/you/terrascope-data`.
To try the built-in sample project, also set `TERRASCOPE_DEV_MODE=1` (never do that on a shared server).

## 4. Build and start
```
docker compose up --build
```
The first run takes several minutes. When you see lines like `Uvicorn running on http://0.0.0.0:8000` and the worker starting, it is up.
Leave this terminal open. (To run in the background use `docker compose up --build -d`; stop with `docker compose down`.)

## 5. Open it
Go to **http://localhost:8000** in your browser. With dev mode on, click "Add synthetic sample project" and open Terrain 3D to confirm everything works.

## 6. Use it with real photos
1. Create a project (enter your EPSG code, or accept the suggestion from the photos' GPS).
2. Upload tab: drop the drone photos (start with 30-50 for a first test).
3. Click "Classify drone photos" if you want per-photo land cover.
4. Dashboard: run photogrammetry (OpenDroneMap) with the **draft** preset first. The first run downloads the large OpenDroneMap image.
5. When it finishes, open Terrain 2D to run terrain reconstruction, then Terrain 3D, then "Edit points".

## If something goes wrong
* **`TERRASCOPE_DATA_DIR` error from compose:** `.env` is missing or the variable is empty.
* **Build fails while installing packages:** copy the last 30 lines and send them to me.
* **Photogrammetry job fails immediately:** open the job's Logs. The usual causes are Docker not reachable from the worker (the compose file mounts `/var/run/docker.sock`; on Docker Desktop this works inside WSL/Mac), the data path not being absolute, or too little memory for OpenDroneMap (give Docker Desktop more RAM in its settings).
* **Port 8000 already in use:** change `"8000:8000"` in `docker-compose.yml` to e.g. `"8080:8000"` and browse to http://localhost:8080.
* **Security:** the worker has access to the Docker socket, which is root-equivalent on your machine, and the app has no login. Run it only on a computer and network you trust.
