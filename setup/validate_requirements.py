import os
import re
import sys
import shutil
import argparse
import setup_common

# Получить абсолютный путь к каталогу текущего файла (каталог проекта)
project_directory = os.path.dirname(os.path.abspath(__file__))

# Проверка, присутствует ли каталог «setup» в каталоге проекта.
if "setup" in project_directory:
    # Если каталог «setup» присутствует, переместитесь на один уровень выше в родительский каталог.
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Добавление каталога проекта в начало пути поиска Python.
sys.path.insert(0, project_directory)

from src.utils.custom_logging import setup_logging

# Настройка ведения журнала
log = setup_logging()


def check_tensorflow():
    # Проверка наличия CUDA или ROCm
    if shutil.which('nvidia-smi') is not None or os.path.exists(
            os.path.join(
                os.environ.get('SystemRoot') or r'C:\Windows',
                'System32',
                'nvidia-smi.exe',
            )
    ):
        log.info('nVidia toolkit обнаружен')
    elif shutil.which('rocminfo') is not None or os.path.exists(
            '/opt/rocm/bin/rocminfo'
    ):
        log.info('AMD toolkit обнаружен')
    else:
        log.info('Подключение только CPU TensorFlow')

    try:
        # Проверка наличия TensorFlow
        import tensorflow as tf

        log.info(f'TensorFlow {tf.__version__}')

        # Проверка доступности GPU
        if not tf.config.list_physical_devices('GPU'):
            log.warning('TensorFlow сообщил, что GPU не доступен')
        else:
            # Информация о доступных устройствах
            for index, device in enumerate(tf.config.list_physical_devices('GPU')):
                log.info(f'TensorFlow обнаружил устройство: {device.name}')

                # Получение информации о GPU
                if 'GPU' in device.name:
                    details = tf.config.experimental.get_device_details(device)
                    log.info(f"Имя устройства: {details.get('device_name')}")
        return 2

    except Exception as e:
        log.error(f'Невозможно загрузить tensorflow: {e}')
        return 0


def main():
    setup_common.check_repo_version()
    # Разобрать аргументы командной строки
    parser = argparse.ArgumentParser(
        description='Validate that requirements are satisfied.'
    )
    parser.add_argument(
        '-r',
        '--requirements',
        type=str,
        help='Path to the requirements file.',
    )
    parser.add_argument('--debug', action='store_true', help='Debug on')
    args = parser.parse_args()

    tensorflow_ver = check_tensorflow()
    setup_common.install_requirements('requirements.txt', check_no_verify_flag=True)

    if args.requirements:
        setup_common.install_requirements(args.requirements, check_no_verify_flag=True)
    else:
        if tensorflow_ver == 0:
            setup_common.install_requirements('requirements_windows_tensorflow.txt', check_no_verify_flag=True)


if __name__ == '__main__':
    main()
