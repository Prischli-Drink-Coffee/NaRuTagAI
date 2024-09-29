import platform
import subprocess
import os

# Информация системы
system = platform.system()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

# Вывод системной информации
print("Системная информация:")
print(f"Система: {system}, Выпуск: {release}, Версия: {version}, Машина: {machine}, Процессор: {processor}")

# Инормация версии python
python_version = platform.python_version()
python_implementation = platform.python_implementation()
python_compiler = platform.python_compiler()

# Вывод версии python
print("\nPython информация:")
print(f"Версия: {python_version}, Реализация: {python_implementation}, Компилятор: {python_compiler}")

# Информация версии виртуальной среды
venv = os.environ.get('VIRTUAL_ENV', None)

# Вывод версии виртуальной среды
if venv:
    print("\nИнформация виртуальной среды:")
    print(f"Path: {venv}")
else:
    print("\nИнформация виртуальной среды:")
    print("Не запускается внутри виртуальной среды.")

# Информация GPU (требует установки nvidia-smi)
try:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])
    output = output.decode('utf-8').strip().split('\n')[1:]
    gpu_info = [line.split(', ') for line in output]
    gpu_name, gpu_vram = gpu_info[0]
    gpu_vram = gpu_vram.replace(' MiB', '')
except (subprocess.CalledProcessError, FileNotFoundError):
    gpu_name, gpu_vram = "N/A", "N/A"
    gpu_vram_warning = False

# Вывод GPU информации
print("\nGPU информация:")
print(f"Имя: {gpu_name}, Оперативная память: {gpu_vram} MiB")

print(' ')
