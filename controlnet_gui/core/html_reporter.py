"""
HTML 报告导出模块
生成包含统计图表和样本预览的可视化筛选结果报告
"""

import os
import json
import base64
import datetime
from typing import List, Optional
from io import BytesIO


def _encode_image_to_base64(image_path: str, max_size: int = 200) -> Optional[str]:
    """将图片编码为 base64 字符串，缩放到指定最大尺寸"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=75)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


def _score_to_color(score: float) -> str:
    """分数转颜色（10分制）"""
    if score >= 8.0:
        return '#4CAF50'
    elif score >= 6.0:
        return '#FFC107'
    elif score >= 4.0:
        return '#FF9800'
    else:
        return '#F44336'


def _build_css() -> str:
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0d0d0d;
            color: #e0e0e0;
            padding: 24px;
        }
        h1 { color: #4CAF50; font-size: 2em; margin-bottom: 8px; }
        h2 { color: #90CAF9; font-size: 1.3em; margin: 32px 0 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }
        h3 { color: #ccc; font-size: 1em; margin-bottom: 8px; }
        .meta { color: #888; font-size: 0.85em; margin-bottom: 24px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .stat-card .value { font-size: 2em; font-weight: bold; }
        .stat-card .label { color: #888; font-size: 0.85em; margin-top: 4px; }
        .bar-chart { margin: 16px 0 24px; }
        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 10px;
        }
        .bar-label { width: 80px; text-align: right; color: #aaa; font-size: 0.85em; flex-shrink: 0; }
        .bar-track {
            flex: 1;
            background: #222;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .bar-count { width: 50px; color: #ccc; font-size: 0.85em; flex-shrink: 0; }
        .samples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
        }
        .sample-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        .sample-card .thumb-row {
            display: flex;
            gap: 2px;
            background: #111;
        }
        .sample-card .thumb-row img {
            flex: 1;
            min-width: 0;
            object-fit: cover;
            height: 80px;
        }
        .sample-card .thumb-row .no-thumb {
            flex: 1;
            height: 80px;
            background: #1a1a1a;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #444;
            font-size: 0.75em;
        }
        .sample-card .info {
            padding: 8px 10px;
        }
        .sample-card .filename {
            font-size: 0.78em;
            color: #90CAF9;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 4px;
        }
        .score-badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: bold;
            color: #000;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 4px;
        }
        .status-accept { background: #1b5e20; color: #A5D6A7; }
        .status-reject { background: #b71c1c; color: #FFCDD2; }
        .status-review { background: #e65100; color: #FFE0B2; }
        .score-dist { margin-bottom: 24px; }
    """


def _build_stat_card(value, label: str, color: str = '#4CAF50') -> str:
    return f"""
        <div class="stat-card">
            <div class="value" style="color:{color}">{value}</div>
            <div class="label">{label}</div>
        </div>"""


def _build_bar_row(label: str, count: int, total: int, color: str) -> str:
    pct = (count / total * 100) if total > 0 else 0
    return f"""
        <div class="bar-row">
            <div class="bar-label">{label}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            <div class="bar-count">{count} ({pct:.0f}%)</div>
        </div>"""


def _build_sample_card(row_data: dict, max_thumb_size: int = 200) -> str:
    """为一行数据构建样本卡片 HTML"""
    variants = row_data.get('variants', [])
    image_path = row_data.get('image_path', '')
    basename = os.path.basename(image_path) if image_path else '未知'
    status = row_data.get('_report_status', 'review')

    status_html = {
        'accept': '<span class="status-badge status-accept">已接受</span>',
        'reject': '<span class="status-badge status-reject">已拒绝</span>',
        'review': '<span class="status-badge status-review">待审核</span>',
    }.get(status, '')

    # 缩略图行（原图 + 各变体）
    thumbs_html = ''
    # 原图
    orig_b64 = _encode_image_to_base64(image_path, max_thumb_size) if image_path and os.path.exists(image_path) else None
    if orig_b64:
        thumbs_html += f'<img src="data:image/jpeg;base64,{orig_b64}" title="原图" />'
    else:
        thumbs_html += '<div class="no-thumb">原图</div>'

    # 最佳变体
    best_score = 0.0
    best_score_str = '-'
    for v in variants:
        score = v.get('score_10', 0) or 0
        ctrl_path = v.get('control_path', '')
        if ctrl_path and os.path.exists(ctrl_path):
            v_b64 = _encode_image_to_base64(ctrl_path, max_thumb_size)
            if v_b64:
                thumbs_html += f'<img src="data:image/jpeg;base64,{v_b64}" title="控制图 score={score:.1f}" />'
            else:
                thumbs_html += f'<div class="no-thumb">{v.get("control_type", "?")}\n{score:.1f}</div>'
        if score > best_score:
            best_score = score
            best_score_str = f'{score:.1f}'

    score_color = _score_to_color(best_score)

    return f"""
        <div class="sample-card" style="border-top:2px solid {score_color}">
            <div class="thumb-row">{thumbs_html}</div>
            <div class="info">
                <div class="filename" title="{image_path}">{basename}</div>
                <span class="score-badge" style="background:{score_color}">{best_score_str}</span>
                {status_html}
            </div>
        </div>"""


def generate_html_report(
    stats: dict,
    rows: list,
    output_path: str,
    title: str = 'ControlNet 数据集筛选报告',
    max_samples: int = 60,
    max_thumb_size: int = 180,
) -> str:
    """
    生成 HTML 报告文件。

    Args:
        stats: 统计字典，包含 total/auto_accept/auto_reject/need_review
        rows:  ImageRowWidget._data 列表，每项需有 _report_status 字段
        output_path: 输出文件路径（.html）
        title: 报告标题
        max_samples: 最多展示的样本数
        max_thumb_size: 缩略图最大边长（像素）

    Returns:
        写入的文件路径
    """
    total = stats.get('total', 0)
    accepted = stats.get('auto_accept', 0)
    rejected = stats.get('auto_reject', 0)
    review = stats.get('need_review', 0)
    generated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- 统计卡片 ----
    stat_cards = (
        _build_stat_card(total, '总处理数', '#90CAF9')
        + _build_stat_card(accepted, '已接受', '#4CAF50')
        + _build_stat_card(rejected, '已拒绝', '#F44336')
        + _build_stat_card(review, '待审核', '#FF9800')
    )

    # ---- 状态分布柱状图 ----
    bar_rows = (
        _build_bar_row('已接受', accepted, total, '#4CAF50')
        + _build_bar_row('已拒绝', rejected, total, '#F44336')
        + _build_bar_row('待审核', review, total, '#FF9800')
    )

    # ---- 分数分布（按 score_10 区间统计）----
    score_buckets = {'9-10': 0, '7-9': 0, '5-7': 0, '3-5': 0, '0-3': 0}
    for row in rows:
        for v in row.get('variants', []):
            s = v.get('score_10', 0) or 0
            if s >= 9:
                score_buckets['9-10'] += 1
            elif s >= 7:
                score_buckets['7-9'] += 1
            elif s >= 5:
                score_buckets['5-7'] += 1
            elif s >= 3:
                score_buckets['3-5'] += 1
            else:
                score_buckets['0-3'] += 1

    score_total = sum(score_buckets.values()) or 1
    score_colors = {'9-10': '#4CAF50', '7-9': '#8BC34A', '5-7': '#FFC107', '3-5': '#FF9800', '0-3': '#F44336'}
    score_bar_rows = ''.join(
        _build_bar_row(k, v, score_total, score_colors[k])
        for k, v in score_buckets.items()
    )

    # ---- 样本预览（优先展示已接受，再审核，再拒绝）----
    accepted_rows = [r for r in rows if r.get('_report_status') == 'accept']
    review_rows = [r for r in rows if r.get('_report_status') == 'review']
    rejected_rows = [r for r in rows if r.get('_report_status') == 'reject']

    sample_pool = (accepted_rows[:max_samples // 2]
                   + review_rows[:max_samples // 4]
                   + rejected_rows[:max_samples // 4])[:max_samples]

    samples_html = ''.join(_build_sample_card(r, max_thumb_size) for r in sample_pool)
    sample_count = len(sample_pool)

    # ---- 组装 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_build_css()}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">生成时间：{generated_at}</div>

<h2>总体统计</h2>
<div class="stats-grid">{stat_cards}</div>

<h2>处理结果分布</h2>
<div class="bar-chart">{bar_rows}</div>

<h2>分数分布（控制图质量评分 / 10分制）</h2>
<div class="score-dist bar-chart">{score_bar_rows}</div>

<h2>样本预览（共 {sample_count} 张）</h2>
<div class="samples-grid">{samples_html}</div>

</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path
