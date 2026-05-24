#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
python train_models.py
python manage.py collectstatic --no-input
python manage.py migrate
