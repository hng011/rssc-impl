#!/bin/bash

source ./ui/.venv/bin/activate && streamlit run ./ui/streamlit_app.py --server.port=8501 & docker compose up