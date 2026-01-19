import json
import sys
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Compare two lm-eval-harness JSONL evaluation results')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline JSONL file')
    parser.add_argument('--target', type=str, required=True, help='Path to target JSONL file')
    parser.add_argument('--output', type=str, default='comparison.html', help='Output HTML file path')
    return parser.parse_args()

def read_jsonl(file_path: str) -> List[Dict]:
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_topk_avg(topks_data: Any) -> float:
    """计算topk的flatten后平均值"""
    if not topks_data:
        return 0.0
    
    try:
        import numpy as np
        topks_data = np.array(topks_data)
        flat_topks = topks_data.flatten()
        if len(flat_topks) == 0:
            return 0.0
        return float(np.mean(flat_topks))
    except (TypeError, ValueError):
        pass
    
    return 0.0

def generate_html(baseline_data: List[Dict], target_data: List[Dict], output_path: str):
    """生成HTML比对报告"""
    
    # 确保数据长度一致
    min_len = min(len(baseline_data), len(target_data))
    baseline_data = baseline_data[:min_len]
    target_data = target_data[:min_len]
    
    # 读取HTML模板
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LM Evaluation Results Comparison</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            padding: 20px;
        }
        
        .container {
            max-width: 95%;
            margin: 0 auto;
        }
        
        header {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        header h1 {
            font-size: 2.2rem;
            margin-bottom: 10px;
        }
        
        header .stats {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .stat-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 8px;
            min-width: 150px;
        }
        
        .controls {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }
        
        .column-selector {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        
        .controls h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.4rem;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 8px;
        }
        
        .search-box {
            margin-top: 15px;
        }
        
        .search-box input {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .comparison-table th {
            background-color: #2c3e50;
            color: white;
            padding: 18px 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .comparison-table td {
            padding: 15px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: top;
        }
        
        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        
        .doc-id {
            font-weight: bold;
            color: #6a11cb;
            text-align: center;
        }
        
        .question {
            max-width: 400px;
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        
        .response {
            max-width: 300px;
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        
        .baseline-response {
            border-left: 4px solid #e74c3c;
        }
        
        .target-response {
            border-left: 4px solid #2ecc71;
        }
        
        .score {
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            padding: 8px;
            border-radius: 6px;
        }
        
        .score-baseline {
            background-color: rgba(231, 76, 60, 0.1);
            color: #c0392b;
        }
        
        .score-target {
            background-color: rgba(46, 204, 113, 0.1);
            color: #27ae60;
        }
        
        .score-diff {
            font-weight: bold;
            text-align: center;
            padding: 8px;
            border-radius: 6px;
        }
        
        .score-improved {
            background-color: rgba(46, 204, 113, 0.2);
            color: #27ae60;
        }
        
        .score-worsened {
            background-color: rgba(231, 76, 60, 0.2);
            color: #c0392b;
        }
        
        .score-same {
            background-color: rgba(241, 196, 15, 0.2);
            color: #f39c12;
        }
        
        .topk-avg {
            text-align: center;
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 6px;
        }
        
        .context {
            max-width: 500px;
            background-color: #f1f8ff;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #9b59b6;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .highlight {
            background-color: #fffacd;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .filter {
            text-align: center;
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        .match-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .match-true {
            background-color: #2ecc71;
        }
        
        .match-false {
            background-color: #e74c3c;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        @media (max-width: 1200px) {
            .container {
                max-width: 100%;
            }
            
            .column-selector {
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            }
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .comparison-table th,
            .comparison-table td {
                padding: 10px 8px;
                font-size: 0.9rem;
            }
            
            .column-selector {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-balance-scale"></i> LM Evaluation Results Comparison</h1>
            <p>Baseline vs Target Model Performance Analysis</p>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Total Samples</div>
                    <div class="stat-value" id="total-samples">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Baseline Avg Score</div>
                    <div class="stat-value" id="baseline-avg">0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Target Avg Score</div>
                    <div class="stat-value" id="target-avg">0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Improvement</div>
                    <div class="stat-value" id="improvement">0.00</div>
                </div>
            </div>
        </header>
        
        <div class="controls">
            <h2><i class="fas fa-sliders-h"></i> Display Controls</h2>
            <div class="search-box">
                <input type="text" id="search-input" placeholder="Search in questions, responses, or by doc_id...">
            </div>
            
            <h3 style="margin-top: 20px; margin-bottom: 10px;">Visible Columns:</h3>
            <div class="column-selector" id="column-selector">
                <!-- Checkboxes will be inserted here by JavaScript -->
            </div>
        </div>
        
        <div style="overflow-x: auto;">
            <table class="comparison-table" id="comparison-table">
                <thead>
                    <tr id="table-header">
                        <!-- Header will be populated by JavaScript -->
                    </tr>
                </thead>
                <tbody id="table-body">
                    <!-- Rows will be populated by JavaScript -->
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by LM Evaluation Comparison Tool | <span id="timestamp"></span></p>
        </div>
    </div>
    
    <script>
        // Data will be injected here
        const comparisonData = {comparison_data};
        
        // Column configuration
        const columns = [
            { id: 'doc_id', name: 'ID', visible: true, type: 'number' },
            { id: 'question', name: 'Question', visible: true, type: 'text' },
            { id: 'context', name: 'Context', visible: true, type: 'text' },
            { id: 'baseline_response', name: 'Baseline Response', visible: true, type: 'text' },
            { id: 'target_response', name: 'Target Response', visible: true, type: 'text' },
            { id: 'baseline_score', name: 'Baseline Score', visible: true, type: 'number' },
            { id: 'target_score', name: 'Target Score', visible: true, type: 'number' },
            { id: 'score_diff', name: 'Score Diff', visible: true, type: 'number' },
            { id: 'baseline_filtered', name: 'Baseline Filtered', visible: true, type: 'text' },
            { id: 'target_filtered', name: 'Target Filtered', visible: true, type: 'text' },
            { id: 'baseline_topk', name: 'Baseline TopK Avg', visible: true, type: 'number' },
            { id: 'target_topk', name: 'Target TopK Avg', visible: true, type: 'number' },
            { id: 'filter_type', name: 'Filter Type', visible: false, type: 'text' },
            { id: 'exact_match', name: 'Exact Match', visible: false, type: 'boolean' }
        ];
        
        // Initialize the table
        function initTable() {
            updateStats();
            renderColumnSelector();
            renderTableHeader();
            renderTableBody();
            setupSearch();
        }
        
        // Update statistics
        function updateStats() {
            const totalSamples = comparisonData.length;
            const baselineAvg = comparisonData.reduce((sum, row) => sum + row.baseline_score, 0) / totalSamples;
            const targetAvg = comparisonData.reduce((sum, row) => sum + row.target_score, 0) / totalSamples;
            const improvement = targetAvg - baselineAvg;
            
            document.getElementById('total-samples').textContent = totalSamples;
            document.getElementById('baseline-avg').textContent = baselineAvg.toFixed(4);
            document.getElementById('target-avg').textContent = targetAvg.toFixed(4);
            document.getElementById('improvement').textContent = improvement.toFixed(4);
            
            // Set timestamp
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleString();
        }
        
        // Render column selector checkboxes
        function renderColumnSelector() {
            const container = document.getElementById('column-selector');
            container.innerHTML = '';
            
            columns.forEach(col => {
                const checkboxId = `col-${col.id}`;
                const div = document.createElement('div');
                div.className = 'checkbox-group';
                div.innerHTML = `
                    <input type="checkbox" id="${checkboxId}" ${col.visible ? 'checked' : ''}>
                    <label for="${checkboxId}">${col.name}</label>
                `;
                
                div.querySelector('input').addEventListener('change', (e) => {
                    col.visible = e.target.checked;
                    renderTableHeader();
                    renderTableBody();
                });
                
                container.appendChild(div);
            });
        }
        
        // Render table header
        function renderTableHeader() {
            const headerRow = document.getElementById('table-header');
            headerRow.innerHTML = '';
            
            columns.forEach(col => {
                if (col.visible) {
                    const th = document.createElement('th');
                    th.textContent = col.name;
                    th.dataset.column = col.id;
                    headerRow.appendChild(th);
                }
            });
        }
        
        // Render table body
        function renderTableBody() {
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            comparisonData.forEach(row => {
                const tr = document.createElement('tr');
                
                columns.forEach(col => {
                    if (!col.visible) return;
                    
                    const td = document.createElement('td');
                    td.dataset.column = col.id;
                    
                    switch(col.id) {
                        case 'doc_id':
                            td.className = 'doc-id';
                            td.textContent = row.doc_id;
                            break;
                            
                        case 'question':
                            td.innerHTML = `<div class="question">${escapeHtml(row.question)}</div>`;
                            break;
                            
                        case 'context':
                            td.innerHTML = `<div class="context">${escapeHtml(row.context)}</div>`;
                            break;
                            
                        case 'baseline_response':
                            td.innerHTML = `<div class="response baseline-response">${escapeHtml(row.baseline_response)}</div>`;
                            break;
                            
                        case 'target_response':
                            td.innerHTML = `<div class="response target-response">${escapeHtml(row.target_response)}</div>`;
                            break;
                            
                        case 'baseline_score':
                            td.innerHTML = `<div class="score score-baseline">${row.baseline_score.toFixed(4)}</div>`;
                            break;
                            
                        case 'target_score':
                            td.innerHTML = `<div class="score score-target">${row.target_score.toFixed(4)}</div>`;
                            break;
                            
                        case 'score_diff':
                            const diff = row.target_score - row.baseline_score;
                            let diffClass = 'score-same';
                            if (diff > 0) diffClass = 'score-improved';
                            else if (diff < 0) diffClass = 'score-worsened';
                            
                            td.innerHTML = `<div class="score-diff ${diffClass}">${diff > 0 ? '+' : ''}${diff.toFixed(4)}</div>`;
                            break;
                            
                        case 'baseline_filtered':
                        case 'target_filtered':
                            td.innerHTML = `<div class="filter">${escapeHtml(row[col.id])}</div>`;
                            break;
                            
                        case 'baseline_topk':
                        case 'target_topk':
                            td.innerHTML = `<div class="topk-avg">${row[col.id].toFixed(4)}</div>`;
                            break;
                            
                        case 'filter_type':
                            td.textContent = row.filter_type;
                            break;
                            
                        case 'exact_match':
                            const matchClass = row.exact_match ? 'match-true' : 'match-false';
                            td.innerHTML = `<span class="match-indicator ${matchClass}"></span>${row.exact_match ? 'Match' : 'No Match'}`;
                            break;
                            
                        default:
                            td.textContent = row[col.id] || '';
                    }
                    
                    tr.appendChild(td);
                });
                
                tbody.appendChild(tr);
            });
        }
        
        // Setup search functionality
        function setupSearch() {
            const searchInput = document.getElementById('search-input');
            searchInput.addEventListener('input', (e) => {
                const searchTerm = e.target.value.toLowerCase().trim();
                
                if (!searchTerm) {
                    // Show all rows
                    document.querySelectorAll('#table-body tr').forEach(tr => {
                        tr.style.display = '';
                    });
                    return;
                }
                
                document.querySelectorAll('#table-body tr').forEach(tr => {
                    const rowText = tr.textContent.toLowerCase();
                    tr.style.display = rowText.includes(searchTerm) ? '' : 'none';
                });
            });
        }
        
        // Utility function to escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initTable);
    </script>
</body>
</html>"""
    
    # 准备对比数据
    comparison_data = []
    
    for i, (baseline, target) in enumerate(zip(baseline_data, target_data)):
        # 提取公共上下文（few-shot示例部分）
        context_parts = []
        if "arguments" in baseline and "gen_args_0" in baseline["arguments"]:
            args = baseline["arguments"]["gen_args_0"]
            if "arg_0" in args:
                # 提取few-shot示例部分（去除当前问题）
                context = args["arg_0"]
                # 移除当前问题及其之后的文本
                current_q = baseline["doc"]["question"]
                q_index = context.find(f"Q: {current_q}")
                if q_index != -1:
                    context = context[:q_index].strip()
                context_parts.append(context)
        
        # 合并上下文
        context_text = "\n\n".join(context_parts) if context_parts else "No context available"
        
        # 获取响应
        baseline_resp = baseline["resps"][0][0] if baseline["resps"] and baseline["resps"][0] else "No response"
        target_resp = target["resps"][0][0] if target["resps"] and target["resps"][0] else "No response"
        
        # 获取过滤后的响应
        baseline_filtered = baseline.get("filtered_resps", ["N/A"])[0] if baseline.get("filtered_resps") else "N/A"
        target_filtered = target.get("filtered_resps", ["N/A"])[0] if target.get("filtered_resps") else "N/A"
        
        # 获取分数
        baseline_score = baseline.get("exact_match", 0.0)
        target_score = target.get("exact_match", 0.0)
        
        # 计算topk平均值
        baseline_topk_avg = calculate_topk_avg(baseline.get("topks", []))
        target_topk_avg = calculate_topk_avg(target.get("topks", []))
        
        comparison_data.append({
            "doc_id": baseline.get("doc_id", i),
            "question": baseline["doc"]["question"],
            "context": context_text,
            "baseline_response": baseline_resp,
            "target_response": target_resp,
            "baseline_filtered": baseline_filtered,
            "target_filtered": target_filtered,
            "baseline_score": float(baseline_score),
            "target_score": float(target_score),
            "baseline_topk": baseline_topk_avg,
            "target_topk": target_topk_avg,
            "filter_type": baseline.get("filter", "N/A"),
            "exact_match": bool(baseline.get("exact_match", 0))
        })
    
    # 将数据注入到HTML模板中
    html_content = html_template.replace(
        'const comparisonData = {comparison_data};',
        f'const comparisonData = {json.dumps(comparison_data, indent=2)};'
    )
    
    # 写入HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML comparison report generated: {output_path}")
    print(f"Total samples compared: {len(comparison_data)}")

def main():
    args = parse_args()
    
    # 检查文件是否存在
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)
    
    if not Path(args.target).exists():
        print(f"Error: Target file not found: {args.target}")
        sys.exit(1)
    
    # 读取数据
    print(f"Reading baseline data from: {args.baseline}")
    baseline_data = read_jsonl(args.baseline)
    
    print(f"Reading target data from: {args.target}")
    target_data = read_jsonl(args.target)
    
    print(f"Baseline samples: {len(baseline_data)}")
    print(f"Target samples: {len(target_data)}")
    
    # 生成HTML报告
    generate_html(baseline_data, target_data, args.output)

if __name__ == "__main__":
    main()
