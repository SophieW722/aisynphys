import subprocess

print(__file__)


def init_colab():
    """Install extra packages if running in CoLab.

    """
    if not in_colab():
        return

    branch = 'release-2-pre-4'
    packages = [
        'git+https://github.com/AllenInstitute/aisynphys@' + branch,
        'git+https://github.com/AllenInstitute/neuroanalysis',
        'lmfit',
    ]

    run('pip install ' + ' '.join(packages))


def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def run(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    while True:
        line = proc.stdout.readline()
        if line == '':
            return
        print(line.decode())


init_colab()