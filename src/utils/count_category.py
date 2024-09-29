from os import listdir
from src.utils.config_parser import ConfigParser


def count_category():
    # Путь к файлу с категориями
    path = os.path.join(path_to_project(), 'category.yaml')
    # Если файл не существует, вызываем исключение
    if not os.path.exists(path):
        raise FileNotFoundError(f'File "{path}" not exists')
    # Считываем YAML файл
    config = ConfigParser.parse(file)
    # Преобразуем список категорий в словарь с нулями
    categories = config.get('Categories', [])
    # Создаем словарь, где каждому ключу соответствует значение 0
    category_count = {category: 0 for category in categories}
    return category_count


dir_ = count_category()
print(list(dir_.keys()))
