import paramiko
import sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting to 158.220.102.37...")
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        safe_line = line.encode("ascii", "ignore").decode("ascii")
        print(safe_line, end="", flush=True)
    return stdout.channel.recv_exit_status()

# 1. Make scripts executable
run_cmd("chmod +x /bot/scripts/*.sh")

# 2. Run vps_setup.sh
print("\n--- Running VPS Provisioning ---")
run_cmd("bash /bot/scripts/vps_setup.sh")

# 3. Launch Docker Compose
print("\n--- Starting Docker Containers ---")
run_cmd("export PATH=$PATH:/usr/bin:/usr/local/bin && cd /bot && docker compose up --build -d")

# 4. Show PS
print("\n--- Docker Status ---")
run_cmd("export PATH=$PATH:/usr/bin:/usr/local/bin && cd /bot && docker compose ps")

ssh.close()
