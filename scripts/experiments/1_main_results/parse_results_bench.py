#!/usr/bin/env python3
import sys
import re
import json
import csv
import argparse
from pathlib import Path
from collections import Counter

def parse_log(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return None, f"Read Error: {str(e)}"

    # 定位 Benchmark 区块
    pattern = r"============ Serving Benchmark Result ============(.*?)=================================================="
    block_match = re.search(pattern, content, re.DOTALL)
    
    if not block_match:
        return None, "Format Error: Could not find benchmark result block."

    block = block_match.group(1)
    data = {"filename": str(file_path)}
    
    lines = block.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-'): continue
        
        # 处理 Per-position acceptance 列表
        pos_match = re.match(r"Position (\d+):\s+([\d.]+)", line)
        if pos_match:
            data[f"Pos_{pos_match.group(1)}_Acceptance(%)"] = pos_match.group(2)
            continue

        # 解析 Key: Value
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            val = parts[1].strip()
            if val:
                data[key] = val
        else:
            # 处理没有冒号但有固定间距的情况
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 2:
                data[parts[0].strip()] = parts[1].strip()

    return data, None

def main():
    parser = argparse.ArgumentParser(description="Collect Benchmark Data")
    parser.add_argument("files", nargs='+', help="Log files to parse")
    parser.add_argument("--format", choices=['csv', 'json'], default='csv', help="Output format")
    args = parser.parse_args()

    results = []
    errors = []
    all_keys = set()

    # 第一轮解析：收集数据和所有的列名
    for f_path in args.files:
        res, err = parse_log(f_path)
        if err:
            errors.append((f_path, err))
        else:
            results.append(res)
            all_keys.update(res.keys())

    if not results:
        print(f"Error: No valid data extracted from {len(args.files)} files.", file=sys.stderr)
        if errors:
            print("\nErrors encountered:", file=sys.stderr)
            for f, e in errors: print(f"  [{f}]: {e}", file=sys.stderr)
        sys.exit(1)

    # 第二轮：检查字段缺失 (Warnings)
    warnings = []
    for res in results:
        missing = all_keys - set(res.keys())
        if missing:
            warnings.append((res['filename'], missing))

    # 输出主数据到 stdout
    if args.format == 'json':
        print(json.dumps(results, indent=4))
    else:
        fieldnames = sorted(list(all_keys))
        # 确保 filename 始终在第一列
        if "filename" in fieldnames:
            fieldnames.insert(0, fieldnames.pop(fieldnames.index("filename")))
        
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 输出报告到 stderr (不会被重定向到文件)
    print("\n" + "="*30, file=sys.stderr)
    print("ANALYSIS SUMMARY", file=sys.stderr)
    print("="*30, file=sys.stderr)
    print(f"Total files processed: {len(args.files)}", file=sys.stderr)
    print(f"Successful:            {len(results)}", file=sys.stderr)
    print(f"Failed:                {len(errors)}", file=sys.stderr)

    if errors:
        print("\n[ERRORS]", file=sys.stderr)
        for f, e in errors:
            print(f"  - {f}: {e}", file=sys.stderr)

    if warnings:
        print("\n[WARNINGS - Missing Fields]", file=sys.stderr)
        for f, m in warnings:
            print(f"  - {f}: missing {list(m)}", file=sys.stderr)
    print("="*30, file=sys.stderr)

if __name__ == "__main__":
    main()
