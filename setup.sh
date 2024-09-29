#!/bin/bash

PYTHON_VER=3.9

# Проверка соответствия версии Python с рекомендуемой версией.
python_version=$(python --version 2>&1)
if [[ $python_version != "Python $PYTHON_VER"* ]]; then
    echo "Warning: Version Python $PYTHON_VER recommended."
fi

# Создание виртуальной среды, если она не существует.
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

# Создание папки logs/setup, если она не существует.
mkdir -p "./logs/setup"

# Создание папки public, если она не существует.
mkdir -p "./public"

# Создание фала temp, если он не существует.
touch "./temp.txt"

# Деактивация активной среды.
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Вызов внешней программы Python для проверки локальных модулей.
python3 ./setup/check_local_modules.py


# Активация виртуальной среды
source ./venv/bin/activate

# Проверка, как был запущен скрипт.
# shellcheck disable=SC2128
if [[ "$0" != "$BASH_SOURCE" ]]; then
    # Скрипт был запущен в GUI
    echo "Script was run by double-clicking."
    python3 ./setup/setup_windows.py
else
    # Скрипт был запущен из командной строки.
    echo "Script was run from the command line."
    python3 ./setup/setup_windows.py
fi

# Деактивация виртуальной среды
deactivate