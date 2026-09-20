import os
import paramiko

HOST = "158.220.102.37"
USER = "root"
PASS = "7EfOJ1iTE4xjz7KJ5lvCM7ZJPv"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}...")
ssh.connect(HOST, port=22, username=USER, password=PASS)

sftp = ssh.open_sftp()
local_cloud_app = os.path.abspath("src/cloud_app.py")
local_compose = os.path.abspath("docker-compose.yml")

print("SFTP put src/cloud_app.py...")
sftp.put(local_cloud_app, "/bot/src/cloud_app.py")

print("SFTP put docker-compose.yml...")
sftp.put(local_compose, "/bot/docker-compose.yml")
sftp.close()

def run_cmd(cmd):
    print(f"\n[SSH] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    print(out + err)

run_cmd("cd /bot && docker compose up -d --force-recreate api-dashboard")
run_cmd("sleep 4 && cd /bot && docker compose logs --tail=30 api-dashboard")

ssh.close()
