import paramiko

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

# Stop host postgres if running
run_cmd("systemctl stop postgresql || true")
run_cmd("systemctl disable postgresql || true")

# Start docker compose stack
run_cmd("cd /bot && docker compose up -d")

# Check status
print("\n--- Container Status ---")
run_cmd("cd /bot && docker compose ps")

ssh.close()
