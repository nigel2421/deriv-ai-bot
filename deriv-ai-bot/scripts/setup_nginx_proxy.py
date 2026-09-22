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
    out = stdout.read().decode('ascii', errors='ignore')
    err = stderr.read().decode('ascii', errors='ignore')
    print(out + err)

# 1. Recreate container
run_cmd("cd /bot && docker compose up -d --force-recreate api-dashboard")

# 2. Wait for startup
run_cmd("sleep 4")

# 3. Check logs
run_cmd("cd /bot && docker compose logs --tail=25 api-dashboard")

# 4. Configure Nginx reverse proxy on port 80 for multi-application support
nginx_conf = """server {
    listen 80 default_server;
    listen [::]:80 default_server;

    server_name _;

    # Deriv AI Bot application at /bot/
    location /bot/ {
        proxy_pass http://127.0.0.1:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Prefix /bot;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location = /bot {
        return 301 /bot/;
    }

    # Root location (serves default application)
    location / {
        proxy_pass http://127.0.0.1:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
"""

run_cmd(f"echo '{nginx_conf}' > /etc/nginx/sites-available/deriv-bot")
run_cmd("rm -f /etc/nginx/sites-enabled/*")
run_cmd("ln -sf /etc/nginx/sites-available/deriv-bot /etc/nginx/sites-enabled/deriv-bot")
run_cmd("nginx -t")
run_cmd("systemctl reload nginx")

# 5. Check firewall rules
run_cmd("ufw allow 80/tcp && ufw allow 8080/tcp && ufw reload")

# 6. Test curl
run_cmd("curl -I http://127.0.0.1:8080/status")
run_cmd("curl -I http://127.0.0.1/status")

ssh.close()
