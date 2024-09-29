import subprocess
import os
import re
import sys
import filecmp
import logging
import shutil
import sysconfig
import datetime
import platform
import pkg_resources

errors = 0  # Определение переменной 'errors'
log = logging.getLogger('sd')

# настройка консоли и ведение журнала
def setup_logging(clean=False):

    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install

    console = Console(
        log_time=True,
        log_time_format='%H:%M:%S-%f',
        theme=Theme(
            {
                'traceback.border': 'black',
                'traceback.border.syntax_error': 'black',
                'inspect.value.border': 'black',
            }
        ),
    )
    # logging.getLogger("urllib3").setLevel(logging.ERROR)
    # logging.getLogger("httpx").setLevel(logging.ERROR)

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(
        os.path.dirname(__file__),
        f'../logs/setup/{current_datetime_str}.log',
    )

    # Создание католога, если он еще не существует
    log_directory = os.path.dirname(log_file)
    os.makedirs(log_directory, exist_ok=True)

    level = logging.INFO
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s',
        filename=log_file,
        filemode='a',
        encoding='utf-8',
        force=True,
    )
    log.setLevel(
        logging.DEBUG
    ) 
    pretty_install(console=console)
    traceback_install(
        console=console,
        extra_lines=1,
        width=console.width,
        word_wrap=False,
        indent_guides=False,
        suppress=[],
    )
    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format='%H:%M:%S-%f',
        level=level,
        console=console,
    )
    rh.set_name(level)
    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])
    log.addHandler(rh)


def check_repo_version(): # pylint: disable=unused-argument
    if os.path.exists('.release'):
        with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
            release= file.read()
        
        log.info(f'Версия: {release}')
    else:
        log.debug('Невозможно прочесть release...')


# execute git command
def git(arg: str, folder: str = None, ignore: bool = False):
    
    git_cmd = os.environ.get('GIT', "git")
    result = subprocess.run(f'"{git_cmd}" {arg}', check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors # pylint: disable=global-statement
        errors += 1
        log.error(f'Ошибка запуска git: {folder} / {arg}')
        if 'or stash them' in txt:
            log.error(f'Обнаружены локальные изменения: подробности смотрите в журнале...')
        log.debug(f'Git output: {txt}')

def pip(arg: str, ignore: bool = False, quiet: bool = False, show_stdout: bool = False):
    # arg = arg.replace('>=', '==')
    mim = False
    if arg.find("--mim") != -1:
        mim = True
        arg = arg.replace("--mim", "")
    log.debug(arg)
    log.debug(mim)
    if not quiet:
        log.info(f'Установка пакета: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force", "").replace("  ", " ").strip()}')
    log.debug(f"Запуск pip: {arg}")
    if show_stdout:
        if mim:
            subprocess.run(f'"{sys.executable}" -m mim {arg}', shell=True, check=False, env=os.environ)
        else:
            subprocess.run(f'"{sys.executable}" -m pip {arg}', shell=True, check=False, env=os.environ)
    else:
        if mim:
            result = subprocess.run(f'"{sys.executable}" -m mim {arg}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            result = subprocess.run(f'"{sys.executable}" -m pip {arg}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
        txt = txt.strip()
        if result.returncode != 0 and not ignore:
            global errors # pylint: disable=global-statement
            errors += 1
            if mim:
                log.error(f'Ошибка запуска mim: {arg}')
            else:
                log.error(f'Ошибка запуска pip: {arg}')
            log.debug(f'Pip вывод: {txt}')
        return txt


def installed(package, friendly: str = None):
    
    # Удалить скобки и их содержимое из строки с помощью регулярных выражений,
    # например, diffusers[tensorflow]==2.13.0 becomes diffusers==2.13.0
    package = re.sub(r'\[.*?\]', '', package)

    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('-') and not p.startswith('=')
            ]
            pkgs = [
                p.split('/')[-1] for p in pkgs
            ]   # получить только имя пакета при установке с URL-адреса
        
        for pkg in pkgs:
            if '>=' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('>=')]
            elif '==' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('==')]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            spec = pkg_resources.working_set.by_key.get(pkg_name, None)
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(pkg_name.lower(), None)
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(pkg_name.replace('_', '-'), None)

            if spec is not None:
                version = pkg_resources.get_distribution(pkg_name).version
                log.debug(f'Найденная версия пакета: {pkg_name} {version}')

                if pkg_version is not None:
                    if '>=' in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        log.warning(f'Ошибочная версия пакета: {pkg_name} {version} required {pkg_version}')
                        return False
            else:
                log.debug(f'Версия пакета не найдена: {pkg_name}')
                return False

        return True
    except ModuleNotFoundError:
        log.debug(f'Пакет не установлен: {pkgs}')
        return False


# установить пакет с помощью pip, если он еще не установлен
def install(
    package,
    friendly: str = None,
    ignore: bool = False,
    reinstall: bool = False,
    show_stdout: bool = False,
):
    # Удалить все после '#' в переменной пакета
    package = package.split('#')[0].strip()

    if reinstall:
        global quick_allowed   # pylint: disable=global-statement
        quick_allowed = False
    if reinstall or not installed(package, friendly):
        pip(f'install --upgrade {package}', ignore=ignore, show_stdout=show_stdout)



def process_requirements_line(line, show_stdout: bool = False):
    package_name = re.sub(r'\[.*?\]', '', line)
    install(line, package_name, show_stdout=show_stdout)


def install_requirements(requirements_file, check_no_verify_flag=False, show_stdout: bool = False):
    if check_no_verify_flag:
        log.info(f'Проверка статуса установки модулей из {requirements_file}...')
    else:
        log.info(f'Установка модулей из {requirements_file}...')
    with open(requirements_file, 'r', encoding='utf8') as f:
        # Чтение строки из файла требований, удаление пробелов и фильтрация пустых строк, комментариев и строк, начинающихся с «.».
        if check_no_verify_flag:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() != ''
                and not line.startswith('#')
                and line is not None
                and 'no_verify' not in line
            ]
        else:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() != ''
                and not line.startswith('#')
                and line is not None
            ]

        # Перебрать каждую строку и установить требования
        for line in lines:
            # Проверка, начинается ли строка с «-r», чтобы включить другой файл требований
            if line.startswith('-r'):
                # Поление пути к включенному файлу требований
                included_file = line[2:].strip()
                # Рекурсивно развернуть включенный файл требований
                install_requirements(included_file, check_no_verify_flag=check_no_verify_flag, show_stdout=show_stdout)
            else:
                process_requirements_line(line, show_stdout=show_stdout)


def ensure_base_requirements():
    try:
        import rich   # pylint: disable=unused-import
    except ImportError:
        install('rich', 'rich')


def run_cmd(run_cmd):
    try:
        with subprocess.Popen(run_cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ) as process:
            process.wait()
    except subprocess.CalledProcessError as e:
        print(f'Произошла ошибка при выполнении команды: {run_cmd}')
        print(f'Ошибка: {e}')


# Проверка версии python
def check_python(ignore=True, skip_git=False):

    supported_minors = [9, 10]
    log.info(f'Python {platform.python_version()} on {platform.system()}')
    if not (
        int(sys.version_info.major) == 3
        and int(sys.version_info.minor) in supported_minors
    ):
        log.error(
            f'Несовместимая версия python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.{supported_minors}'
        )
        if not ignore:
            sys.exit(1)
    if not skip_git:
        git_cmd = os.environ.get('GIT', 'git')
        if shutil.which(git_cmd) is None:
            log.error('Git not found')
            if not ignore:
                sys.exit(1)
    else:
        git_version = git('--version', folder=None, ignore=False)
        log.debug(f'Git {git_version.replace("git version", "").strip()}')


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f'Произошла ошибка при записи в файл: {file_path}')
        print(f'Ошибка: {e}')


def clear_screen():
    # Проверьте текущую операционную систему, чтобы выполнить правильную команду очистки экрана.
    if os.name == 'nt':  # Если операционная система Windows
        os.system('cls')
    else:  # Если операционная система Linux или Ma
        os.system('clear')

