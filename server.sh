#!/bin/bash

# Деактивация активной среды, если таковая есть
if [ -f "./venv/bin/activate" ]; then
    deactivate 2>/dev/null
fi

# Установка переменной пути проекта
export PROJECT_PATH="$(pwd)/src"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

# Проверка локальных модулей
python3 ./setup/check_local_modules.py --no_question

# Активация виртуальной среды
source ./venv/bin/activate

# Экспорт пути к библиотекам
# shellcheck disable=SC2155
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/venv/lib/python3.9/site-packages/torch/lib"

# Валидация requirements
python3 ./setup/validate_requirements.py

# Очистка setup.log
python3 ./clear_setup_log.py

# Запуск сервера
python3 ./src/pipeline/server.py