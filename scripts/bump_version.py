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


def adjust_changelog():
    with open('HISTORY.rst', 'r') as file:
        lines = file.readlines()

    # Remove 'Elphick.' prefix from the first line
    prefix = 'Elphick.'
    if lines[0].startswith(prefix):
        lines[0] = lines[0][len(prefix):]

    # Adjust the length of the underline on the second line
    if lines[1].startswith('='):
        lines[1] = '=' * (len(lines[0].strip()) - 1) + '\n'  # -1 for the newline character

    with open('HISTORY.rst', 'w') as file:
        file.writelines(lines)


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

    # remove the news fragments
    run_command("rm -rf newsfragments/*")

    # strip the Elphick. prefix from the top heading only.
    adjust_changelog()


if __name__ == "__main__":
    main()
