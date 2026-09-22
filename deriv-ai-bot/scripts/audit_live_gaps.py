import paramiko
import json

HOST = "158.220.102.37"
USER = "root"
PASS = "7EfOJ1iTE4xjz7KJ5lvCM7ZJPv"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=22, username=USER, password=PASS)

def get_json(url):
    _, stdout, _ = ssh.exec_command(f"curl -s {url}")
    return json.loads(stdout.read().decode('utf-8', errors='ignore'))

di = get_json("http://127.0.0.1:8080/api/v1/decision_intelligence")
diag = get_json("http://127.0.0.1:8080/diag")

print("=== REJECTION FUNNEL ===")
print(json.dumps(di.get("rejection_funnel"), indent=2))

traces = di.get("recent_traces") or []
print(f"\n=== RECENT TRACES (TOTAL {len(traces)}) ===")
from collections import Counter
sym_counts = Counter(t.get("symbol") for t in traces)
print("Traces by Symbol:", dict(sym_counts))
rej_counts = Counter(t.get("rejection_reason") or "EXECUTED" for t in traces)
print("Rejection Reasons:", dict(rej_counts))

print("\n=== LAST 15 TRACES DETAILED ===")
for t in traces[-15:]:
    print(f"Symbol: {t.get('symbol')} | Contract: {t.get('proposed_contract_type')} ({t.get('proposed_duration')}{t.get('proposed_duration_unit')}) | Score: {t.get('consensus_score'):.3f} | HTF: {t.get('htf_alignment_result')} | EV: {t.get('expected_value')} | Verdict: {t.get('final_decision')} | Rej: {t.get('rejection_reason')}")

print("\n=== OFFER GATE BLOCKS ===")
print(json.dumps(diag.get("offer_gate"), indent=2))

print("\n=== RECENT TRADE ERRORS ===")
print(json.dumps(diag.get("recent_trades_errors"), indent=2))

ssh.close()
