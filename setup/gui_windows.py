import sys
from dotenv import load_dotenv, set_key, unset_key
from src import path_to_env
load_dotenv(path_to_env())


def main_menu():
    while True:
        print('=============================================================')
        print(' Меню для работы с NaRuTagAI:\n')
        print(' 1. Запуск data_downloader')
        print(' 2. Запуск data_preproccessor')
        print(' 3. Запуск data_graphcreator')
        print(' 4. Запуск train')
        print(' 5. Запуск uvicorn')
        print(' 6. Запуск pytest')
        print(' 7. Выход из меню')
        print('=============================================================')

        choice = input('\nСделайте выбор: ')
        print('')

        if choice in ['1', '2', '3', '4', '5', '6']:
            set_key(path_to_env(), 'CHOICE', choice)  # Устанавливаем значение переменной окружения
            break
        elif choice == '7':
            sys.exit()
        else:
            print('Выберите между 1-7')


if __name__ == '__main__':
    try:
        main_menu()
    except Exception as ex:
        print(ex)
