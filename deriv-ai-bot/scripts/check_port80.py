import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    print(f"\n=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    print(out + err)

run_cmd("ss -tulpn | grep :80")
run_cmd("lsof -i :80")
run_cmd("systemctl status nginx || systemctl status apache2")

ssh.close()
