import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def check_cmd(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode('utf-8', errors='ignore') + stderr.read().decode('utf-8', errors='ignore')

print("=== PS AUX ===")
print(check_cmd("ps aux | grep -v 'ps aux' | head -n 30"))

print("=== RECENT COMMAND LOGS ===")
print(check_cmd("ls -la /bot/scripts/"))

ssh.close()
