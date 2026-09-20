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
files_to_upload = [
    ("src/cloud_app.py", "/bot/src/cloud_app.py"),
    ("src/orchestrator.py", "/bot/src/orchestrator.py"),
    ("src/strategy/risk_manager.py", "/bot/src/strategy/risk_manager.py"),
    ("docker-compose.yml", "/bot/docker-compose.yml"),
]

for local, remote in files_to_upload:
    local_path = os.path.abspath(local)
    print(f"Uploading {local} -> {remote}")
    sftp.put(local_path, remote)

sftp.close()

def run_cmd(cmd):
    print(f"\n[SSH] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('ascii', errors='ignore')
    err = stderr.read().decode('ascii', errors='ignore')
    print(out + err)

run_cmd("cd /bot && docker compose up -d --force-recreate")
run_cmd("sleep 5")
run_cmd("systemctl reload nginx")
run_cmd("cd /bot && docker compose ps")
run_cmd("curl -I http://127.0.0.1:8080/status")
run_cmd("curl -I http://158.220.102.37/status")
run_cmd("curl -I http://158.220.102.37/")

ssh.close()
