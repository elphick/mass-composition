import argparse
import subprocess
import sys


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()


def process_command_line_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('increment', type=str, help='The increment type (major, minor, patch)')
    args = parser.parse_args()
    return args


def main():
    args = process_command_line_parameters()

    increment = args.increment
    # Validate the input
    if increment not in ["major", "minor", "patch"]:
        print("Invalid version increment. Please enter 'major', 'minor', or 'patch'.")
        sys.exit(1)

    # Run the commands
    run_command(f"poetry version {increment}")
    run_command("poetry install --all-extras")
    # run_command("python towncrier/create_news.py")
    run_command("towncrier")


if __name__ == "__main__":
    main()
