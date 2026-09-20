import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    print(f"\n=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('ascii', errors='ignore')
    err = stderr.read().decode('ascii', errors='ignore')
    print(out + err)

run_cmd("cd /bot && docker compose logs --tail=40 api-dashboard")

ssh.close()
