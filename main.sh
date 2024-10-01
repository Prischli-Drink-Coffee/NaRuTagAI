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

# Меню выбора интерфейса
python3 ./setup/gui_windows.py

# Чтение значения из файла temp.txt
if [ -f "temp.txt" ]; then
    var=$(<temp.txt)
else
    echo "Файл temp.txt не найден."
    exit 1
fi


# Запуск соответствующего скрипта на основе значения в temp.txt
# shellcheck disable=SC2181
if [ $? -eq 0 ]; then
    # Проверка, был ли скрипт запущен двойным кликом или из командной строки
    # shellcheck disable=SC2128
    if [[ "$0" != "$BASH_SOURCE" ]]; then
        echo "Этот скрипт был запущен двойным кликом."

        case "$var" in
            "1")
                gnome-terminal -- python3 ./src/pipeline/data_downloader.py
                ;;
            "2")
                gnome-terminal -- python3 ./src/pipeline/train.py
                ;;
            "3")
                gnome-terminal -- python3 ./src/pipeline/server.py
                ;;
            "4")
                gnome-terminal -- python3 ./src/pipeline/test.py
                ;;
            *)
                echo "Неизвестное значение в temp.txt: $var"
                ;;
        esac

    else
        echo "Этот скрипт был запущен из командной строки."

        case "$var" in
            "1")
                python3 ./src/pipeline/data_downloader.py
                ;;
            "2")
                python3 ./src/pipeline/train.py
                ;;
            "3")
                python3 ./src/pipeline/server.py
                ;;
            "4")
                python3 ./src/pipeline/test.py
                ;;
            *)
                echo "Неизвестное значение в temp.txt: $var"
                ;;
        esac
    fi
fi
