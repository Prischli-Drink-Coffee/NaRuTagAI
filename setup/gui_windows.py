import sys
from dotenv import load_dotenv, set_key, unset_key
from src import path_to_env
from src.utils.custom_logging import setup_logging
load_dotenv(path_to_env())
log = setup_logging()


def main_menu():
    while True:
        log.info('\n=============================================================\n')
        log.info(' Меню для работы с NaRuTagAI:\n')
        log.info(' 1. Запуск data_downloader')
        log.info(' 2. Запуск data_preproccessor')
        log.info(' 3. Запуск data_clustercreator')
        log.info(' 4. Запуск data_graphcreator')
        log.info(' 5. Запуск train')
        log.info(' 6. Запуск train_plotter')
        log.info(' 7. Запуск uvicorn')
        log.info(' 8. Запуск pytest')
        log.info(' 9. Выход из меню')
        log.info('\n=============================================================\n')

        choice = input('\nСделайте выбор: ')
        log.info('')

        if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
            set_key(path_to_env(), 'CHOICE', choice)  # Устанавливаем значение переменной окружения
            break
        elif choice == '9':
            sys.exit()
        else:
            log.info('Выберите между 1-9')


if __name__ == '__main__':
    try:
        main_menu()
    except Exception as ex:
        log.info(ex)
