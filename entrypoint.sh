#!/bin/bash
set -e

# If a marker file doesn't exist in the venv, it means dependencies are not installed.
# Execute the build script to set up the Python environment.
# Your build script should create this marker file upon completion.
if [ ! -f ".venv/.build_complete" ]; then
  echo "Dependencies not installed. Running build script..."
  sudo chown -R dev:dev /app/.venv
  sudo chown -R dev:dev /home/dev/.cache
  sudo chown -R dev:dev /tmp
  /bin/bash scripts/build.sh
  echo "Build script finished."
fi

# Execute the command passed to the container (e.g., bash).
exec "$@"