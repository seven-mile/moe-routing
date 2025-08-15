from IPython.display import HTML, display

def get_ppl_html(tokens, ppl_values):
    """
    根据PPL值为token生成HTML字符串，用于可视化。

    这个函数不直接显示任何内容，而是返回一个完整的HTML字符串。

    参数:
    - tokens (list[str]): token的文本列表。
    - ppl_values (list[float]): 与token对应的PPL值列表。

    返回:
    - str: 一个包含可视化内容的完整HTML字符串。
    """
    if not tokens or not ppl_values or len(tokens) != len(ppl_values):
        raise ValueError("输入列表不能为空，且tokens和ppl_values的长度必须一致。")

    # 1. 找到PPL的最大值和最小值
    min_ppl = min(ppl_values)
    max_ppl = max(ppl_values)
    
    # 处理所有PPL值都相同的情况，避免除以零
    if max_ppl == min_ppl:
        # 如果所有值都一样，给一个中间态的固定透明度
        normalized_ppl_values = [0.5] * len(ppl_values)
    else:
        # 2. PPL值归一化到 0-1 范围
        normalized_ppl_values = [(ppl - min_ppl) / (max_ppl - min_ppl) for ppl in ppl_values]

    # --- HTML和CSS内容构建 ---
    
    # 核心HTML片段，不包含<html>, <body>等标签，方便嵌入
    html_body_content = ""
    
    # 3. 为每个token生成带样式的<span>
    for token, normalized_ppl, original_ppl in zip(tokens, normalized_ppl_values, ppl_values):
        # 基础透明度设为0.1，确保最低PPL也有浅色背景
        # PPL越高，alpha越接近1.0
        alpha = 0.1 + 0.9 * normalized_ppl
        
        # 定义颜色
        blue_color = "30, 144, 255" # RGB for Dodger Blue
        
        # HTML编码token以防特殊字符（如 < >）破坏HTML结构
        safe_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 为每个token创建一个span，并添加一个tooltip来显示精确的PPL值
        span = (
            f'<div class="tooltip">'
            f'<span class="token" style="background-color: rgba({blue_color}, {alpha:.2f});">{safe_token}</span>'
            f'<span class="tooltiptext">PPL: {original_ppl:.2f}</span>'
            f'</div>'
        )
        html_body_content += span + " " # 添加一个空格来分隔token

    # 构建完整的HTML文档（包含CSS样式）
    full_html = f"""
    <style>
        .ppl-vis-container {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 10px; 
            font-size: 1.1em; 
            line-height: 2.2;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        .token {{
            display: inline-block;
            padding: 3px 5px;
            border-radius: 4px;
            margin: 0 1px;
            transition: all 0.2s ease-in-out;
        }}
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: default;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 120px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 130%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
            line-height: 1.4;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .legend {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .gradient-bar {{
            height: 18px;
            width: 100%;
            background: linear-gradient(to right, rgba(30, 144, 255, 0.1), rgba(30, 144, 255, 1.0));
            border-radius: 3px;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: #555;
            margin-top: 4px;
        }}
    </style>
    <div class="ppl-vis-container">
        <div>{html_body_content}</div>
        <div class="legend">
            <div class="legend-title">Perplexity (PPL) Scale</div>
            <div class="gradient-bar"></div>
            <div class="legend-labels">
                <span>Low PPL ({min_ppl:.2f})</span>
                <span>High PPL ({max_ppl:.2f})</span>
            </div>
        </div>
    </div>
    """
    return full_html

def display_ppl_in_notebook(tokens, ppl_values):
    """
    生成PPL可视化并在Jupyter/IPython环境中直接显示。

    参数:
    - tokens (list[str]): token的文本列表。
    - ppl_values (list[float]): 与token对应的PPL值列表。

    返回:
    - IPython.display.HTML: 一个可以被Notebook渲染的HTML对象。
    """
    html_content = get_ppl_html(tokens, ppl_values)
    return HTML(html_content)