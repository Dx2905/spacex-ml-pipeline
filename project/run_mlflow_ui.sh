#!/bin/bash
cd "$(dirname "$0")"  # Automatically go to script's folder
mlflow ui --backend-store-uri file://$(pwd)/mlruns


