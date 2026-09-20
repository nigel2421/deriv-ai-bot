import time
import paramiko

HOST = "158.220.102.37"
USER = "root"
PASS = "7EfOJ1iTE4xjz7KJ5lvCM7ZJPv"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for attempt in range(1, 4):
    try:
        print(f"Connecting to {HOST} (attempt {attempt})...")
        ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=15)
        print("Connected successfully!")
        break
    except Exception as e:
        print(f"Connection error: {e}")
        time.sleep(3)

def run_cmd(cmd):
    print(f"\n=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('ascii', errors='ignore')
    err = stderr.read().decode('ascii', errors='ignore')
    print(out + err)

run_cmd("cd /bot && docker compose ps")
run_cmd("cd /bot && docker compose logs --tail=20 api-dashboard")
run_cmd("curl -I http://127.0.0.1:8080/status")
run_cmd("curl -I http://127.0.0.1/status")
run_cmd("curl -I http://158.220.102.37/")

ssh.close()
