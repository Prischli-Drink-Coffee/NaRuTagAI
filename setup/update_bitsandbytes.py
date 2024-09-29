import os
import sysconfig
import filecmp
import shutil

def sync_bits_and_bytes_files():
    """
    Проверяйте файлы на наличие «разных» битов и байтов и копируйте только при необходимости.
    Эта функция специфична для ОС Windows.
    """
    
    # Выполнять только в Windows
    if os.name != "nt":
        print("Эта функция применима только к ОС Windows...")
        return

    try:
        # Определить исходный и конечный каталоги
        source_dir = os.path.join(os.getcwd(), "bitsandbytes_windows")

        dest_dir_base = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
        
        # Очистить кеш сравнения файлов
        filecmp.clear_cache()
        
        # Перебрать каждый файл в исходном каталоге
        for file in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, file)

            # Определите каталог назначения на основе имени файла
            if file in ("main.py", "paths.py"):
                dest_dir = os.path.join(dest_dir_base, "cuda_setup")
            else:
                dest_dir = dest_dir_base

            # Скопируйте файл из источника в место назначения, сохранив метаданные исходного файла.
            print(f'Copy {source_file_path} to {dest_dir}')
            shutil.copy2(source_file_path, dest_dir)
        print("Процесс завершен успешно")

    except FileNotFoundError as fnf_error:
        print(f"Ошибка файл не найден: {fnf_error}")
    except PermissionError as perm_error:
        print(f"Ошибка прав доступа: {perm_error}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")


if __name__ == "__main__":
    sync_bits_and_bytes_files()