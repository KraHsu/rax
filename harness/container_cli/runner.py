"""Execute DSH shell commands inside its already isolated agent container."""
import os
import sys


def main():
    try:
        supplied = sys.argv[1:]
        command = supplied[supplied.index('--')+1:]
        if not command:
            raise ValueError('Missing sandbox command')
        os.execvp(command[0], command)
    except (OSError, ValueError) as error:
        print(f'RAX_DSH_CONTAINER: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
