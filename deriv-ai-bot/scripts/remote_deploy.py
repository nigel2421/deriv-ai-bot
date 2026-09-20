#!/usr/bin/env python3
"""
Remote deployment script for Contabo VPS using Paramiko SSH & SFTP.
Directly syncs local workspace files to /bot on VPS, installs Docker via vps_setup.sh,
and starts the bot using Docker Compose.
"""

import os
import sys
import paramiko

HOST = "158.220.102.37"
USER = "root"
PASS = "7EfOJ1iTE4xjz7KJ5lvCM7ZJPv"
TARGET_DIR = "/bot"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXCLUDE_DIRS = {"venv", ".git", ".pytest_cache", "__pycache__", ".idea", ".vscode"}
EXCLUDE_FILES = {"*.pyc", "*.pyo"}

def run_ssh_cmd(ssh, cmd, ignore_errors=False):
    print(f"\n[SSH RUN] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    
    for line in iter(stdout.readline, ""):
        print(line, end="")
        
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        if not ignore_errors:
            print(f"[ERROR] Command failed with exit code {exit_code}: {cmd}")
        return False
    return True

def upload_directory(sftp, local_dir, remote_dir):
    print(f"[SFTP] Syncing {local_dir} -> {remote_dir}...")
    
    for root, dirs, files in os.walk(local_dir):
        # Filter excluded directories in-place
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

def main():
    print("================================================================")
    print(f"Connecting to Contabo VPS ({HOST})...")
    print("================================================================")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=15)
        print("[SUCCESS] Connected to SSH server!")
    except Exception as e:
        print(f"[FAILED] SSH Connection failed: {e}")
        sys.exit(1)

    # 1. Prepare target directory on VPS
    run_ssh_cmd(ssh, f"mkdir -p {TARGET_DIR}")

    # 2. Upload all workspace files via SFTP
    print("[INFO] Uploading project files to VPS...")
    sftp = ssh.open_sftp()
    upload_directory(sftp, LOCAL_DIR, TARGET_DIR)
    sftp.close()
    print("[SUCCESS] All project files uploaded.")

    # 3. Make scripts executable and run vps_setup.sh
    print("[INFO] Running vps_setup.sh (Installing Docker, Fail2ban, Firewall)...")
    run_ssh_cmd(ssh, f"chmod +x {TARGET_DIR}/scripts/*.sh")
    run_ssh_cmd(ssh, f"bash {TARGET_DIR}/scripts/vps_setup.sh")

    # 4. Build and run Docker Compose containers
    print("[INFO] Rebuilding and restarting Docker Compose stack...")
    # Force recreate to ensure API dashboard container picks up latest src/cloud_app.py
    cmd_docker = f"export PATH=$PATH:/usr/bin:/usr/local/bin && cd {TARGET_DIR} && docker compose down && docker compose up --build --force-recreate -d"
    run_ssh_cmd(ssh, cmd_docker)


    # 5. Check container status
    print("\n================================================================")
    print("Container Status:")
    print("================================================================")
    run_ssh_cmd(ssh, f"export PATH=$PATH:/usr/bin:/usr/local/bin && cd {TARGET_DIR} && docker compose ps")

    # 6. Check logs summary
    print("\n================================================================")
    print("Recent Logs:")
    print("================================================================")
    run_ssh_cmd(ssh, f"export PATH=$PATH:/usr/bin:/usr/local/bin && cd {TARGET_DIR} && docker compose logs --tail=25")

    ssh.close()
    print("\n================================================================")
    print("DEPLOYMENT COMPLETE!")
    print(f"Live Dashboard URL: http://{HOST}:8080")
    print("================================================================")

if __name__ == "__main__":
    main()
