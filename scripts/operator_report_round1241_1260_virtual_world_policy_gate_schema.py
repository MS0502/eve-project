import json
import sys
from adapters.virtual_world_policy_gate_schema import build_virtual_world_policy_gate_schema_summary

def main():
    summary = build_virtual_world_policy_gate_schema_summary()
    print(json.dumps(summary, separators=(',', ':')))

if __name__ == "__main__":
    main()
