import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('158.220.102.37', username='root', password='7EfOJ1iTE4xjz7KJ5lvCM7ZJPv')

def run_cmd(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode('utf-8', errors='ignore') + stderr.read().decode('utf-8', errors='ignore')

print("=== POSTGRES TRADE MEMORY COUNT ===")
print(run_cmd("docker exec -i bot-postgres psql -U deriv_user -d deriv_trading_db -c 'SELECT count(*) FROM trade_memory;'"))

print("=== WIN/LOSS SUMMARY BY SYMBOL ===")
print(run_cmd("docker exec -i bot-postgres psql -U deriv_user -d deriv_trading_db -c 'SELECT symbol, status, count(*), round(avg(profit::numeric), 4) as avg_pnl FROM trade_memory GROUP BY symbol, status ORDER BY symbol;'"))

print("=== RECENT 20 TRADES ===")
print(run_cmd("docker exec -i bot-postgres psql -U deriv_user -d deriv_trading_db -c 'SELECT symbol, contract_type, confidence, status, profit, market_regime, created_at FROM trade_memory ORDER BY id DESC LIMIT 20;'"))

print("=== RECENT TRADING ENGINE LOGS ===")
print(run_cmd("cd /bot && docker compose logs --tail=40 trading-engine"))

ssh.close()
