import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    print(f"[SSH RUN] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    print(out + err)

print("=== ALLOW PORT 8080 ON UFW ===")
run_cmd("ufw allow 8080/tcp")
run_cmd("ufw reload")

print("=== CHECK FIREWALL STATUS ===")
run_cmd("ufw status verbose")

print("=== CHECK DASHBOARD LOGS ===")
run_cmd("cd /bot && docker compose logs --tail=20 api-dashboard")

ssh.close()
