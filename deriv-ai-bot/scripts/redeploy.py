import os
import sys
import paramiko

HOST = "158.220.102.37"
USER = "root"
PASS = "7EfOJ1iTE4xjz7KJ5lvCM7ZJPv"
TARGET_DIR = "/bot"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXCLUDE_DIRS = {"venv", ".git", ".pytest_cache", "__pycache__", ".idea", ".vscode", "data", "logs", ".system_generated"}

def upload_directory(sftp, local_dir, remote_dir):
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_path = os.path.relpath(root, local_dir)
        if rel_path == ".":
            remote_root = remote_dir
        else:
            remote_root = os.path.normpath(os.path.join(remote_dir, rel_path)).replace("\\", "/")
            
        try:
            sftp.stat(remote_root)
        except IOError:
            sftp.mkdir(remote_root)
            
        for file in files:
            if file.endswith(".pyc"):
                continue
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.normpath(os.path.join(remote_root, file)).replace("\\", "/")
            sftp.put(local_file_path, remote_file_path)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to VPS ({HOST})...")
ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=15)

print("Uploading updated codebase via SFTP...")
sftp = ssh.open_sftp()
upload_directory(sftp, LOCAL_DIR, TARGET_DIR)
sftp.close()
print("Codebase updated on VPS.")

def run_cmd(cmd):
    print(f"\n[SSH RUN] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        safe_line = line.encode("ascii", "ignore").decode("ascii")
        print(safe_line, end="", flush=True)
    return stdout.channel.recv_exit_status()

print("Rebuilding and restarting Docker containers...")
run_cmd(f"cd {TARGET_DIR} && docker compose up --build -d")

print("\n--- Container Status ---")
run_cmd(f"cd {TARGET_DIR} && docker compose ps")

ssh.close()
