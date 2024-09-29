import argparse
import subprocess

# Разбор аргументов командной строки.
parser = argparse.ArgumentParser()
parser.add_argument('--no_question', action='store_true')
args = parser.parse_args()

# Запуск pip freeze и сохранение в переменную output.
output = subprocess.getoutput("pip freeze")

# Удаление строк "WARNING".
output_lines = [line for line in output.splitlines() if "WARNING" not in line]

# Реконструкция output без строк "WARNING".
output = "\n".join(output_lines)

# Проверка наличия модулей в output.
if output:
    print(f"=============================================================")
    print(" Обнаружены модули, установленные вне виртуальной среды.")
    print(" Это может вызвать проблемы. Проверьте установленные модули.")
    print(" Вы можете удалить все локальные модули с помощью:\n")
    print(" deactivate")
    print(" pip freeze > uninstall.txt")
    print(" pip uninstall -y -r uninstall.txt")
    print(f"=============================================================")
    print('')
