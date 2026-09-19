import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    print(f"[SSH RUN] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    print(out + err)

# Upload updated docker-compose.yml
local_dc = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
sftp = ssh.open_sftp()
sftp.put(local_dc, "/bot/docker-compose.yml")
sftp.close()
print("Uploaded updated docker-compose.yml.")

# Make sure permissions on /bot/data exist
run_cmd("mkdir -p /bot/data/logs && chmod -R 777 /bot/data")

# Restart containers
run_cmd("cd /bot && docker compose up -d --force-recreate")
run_cmd("cd /bot && docker compose ps")

ssh.close()
