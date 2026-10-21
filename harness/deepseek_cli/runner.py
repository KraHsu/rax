"""DSH's custom bwrap runner: offline commands with the RAX filesystem boundary.

Run with isolated Python. The native DSH sandbox supplies bwrap-shaped arguments;
the trusted launcher owns the actual profile, so model-selected arguments cannot
expand its mounts or network access.
"""
import json
import os
from pathlib import Path
import sys


def arguments(boundary, command):
    if not command:
        raise ValueError('Missing sandbox command')
    args = [boundary['bwrap'], '--ro-bind', '/', '/', '--dev', '/dev', '--unshare-net',
            '--unshare-pid', '--proc', '/proc', '--tmpfs', '/tmp', '--die-with-parent']
    masked = []
    for item in boundary['hidden']:
        path = Path(item)
        if path.is_dir():
            args += ['--tmpfs', str(path)]
            masked.append(str(path))
        elif path.exists():
            args += ['--ro-bind', '/dev/null', str(path)]
    for path in boundary['reexposed']:
        args += ['--ro-bind', path, path]
    args += ['--bind', boundary['workspace'], boundary['workspace'],
             '--bind', boundary['socket_directory'], boundary['socket_directory']]
    # Populate mount points first, then seal each mask. The workspace and socket
    # are separate child mounts and retain their intended write access.
    for path in masked:
        args += ['--remount-ro', path]
    args += ['--chdir', boundary['workspace'], '--', *command]
    return args


def main():
    try:
        boundary = json.loads(Path(sys.argv[1]).read_text())
        supplied = sys.argv[2:]
        command = supplied[supplied.index('--')+1:]
        args = arguments(boundary, command)
        allowed = ('HOME', 'PATH', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'RAX_MCP_SOCKET',
                   'TMPDIR', 'LANG', 'LC_ALL', 'TERM', 'USER', 'LOGNAME', 'TZ',
                   'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
        env = {key: value for key, value in os.environ.items() if key in allowed}
        os.execve(args[0], args, env)
    except (OSError, ValueError, IndexError, KeyError) as error:
        print(f'RAX_DSH_SANDBOX: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
