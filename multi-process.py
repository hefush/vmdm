import os
import sys
import subprocess
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Control multiple processes running")
    parser.add_argument("command_file", help="File containing commands to execute")
    parser.add_argument("-cpu", type=int, default=3, help="Number of commands to run in parallel, default=3")
    parser.add_argument("-cmd", action="store_true", help="Output the commands but not execute")
    parser.add_argument("-verbose", action="store_true", help="Output information of running progress")
    return parser.parse_args()

def read_commands(file_path):
    commands = []
    with open(file_path, 'r') as file:
        line = ""
        for raw_line in file:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if raw_line.endswith(';'):
                line += raw_line
            else:
                line += raw_line
                commands.append(line)
                line = ""
        if line:
            commands.append(line)
    return commands

def multiprocess(commands, max_cpu, verbose, cmd_only=False):
    if max_cpu <= 0:
        raise ValueError("-cpu must be a positive integer")

    total = len(commands)
    if verbose:
        print(f"\n\tcmd num:  {total}\n\tcpu num:  {max_cpu}\n\n")

    report = {int(total / 10 * i): i * 10 for i in range(1, 11)}

    processes = []
    failures = []
    for i, cmd in enumerate(commands):
        if verbose and i + 1 in report:
            print(f"\tthrow out  {report[i + 1]}%")

        if len(processes) >= max_cpu:
            while len(processes) >= max_cpu:
                for proc, proc_cmd in processes[:]:
                    ret = proc.poll()
                    if ret is None:
                        continue
                    if ret != 0:
                        failures.append((ret, proc_cmd))
                    processes.remove((proc, proc_cmd))
                if len(processes) >= max_cpu:
                    time.sleep(0.2)

        if verbose:
            print(f"Executing: {cmd}")

        if not cmd_only:
            p = subprocess.Popen(cmd, shell=True)
            processes.append((p, cmd))

    for p, cmd in processes:
        ret = p.wait()
        if ret != 0:
            failures.append((ret, cmd))

    if verbose:
        print("\tAll tasks done")

    if failures:
        for ret, cmd in failures:
            print(f"Command failed with exit code {ret}: {cmd}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    commands = read_commands(args.command_file)

    if args.cmd:
        for cmd in commands:
            print(cmd)
    else:
        try:
            sys.exit(multiprocess(commands, args.cpu, args.verbose, args.cmd))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
