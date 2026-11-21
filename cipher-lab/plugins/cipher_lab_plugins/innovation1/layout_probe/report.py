"""Plain Chinese report; only sealed fields and artifact refs are presented."""
from __future__ import annotations

import html
import json
from typing import Any

from .config import FIXED_METHOD, FIXED_NAMESPACE, METHOD, NAMESPACE, SLOTS


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    recipe = saved['recipe']
    method = recipe.get('method')
    fixed = method == FIXED_METHOD
    if (recipe.get('run_id') != saved['run_id'] or method not in (METHOD, FIXED_METHOD)
            or recipe.get('namespace') != (FIXED_NAMESPACE if fixed else NAMESPACE)
            or summary.get('method', method) != method
            or (saved['status'] == 'SUCCEEDED' and summary.get('method') != method)):
        raise ValueError('LAYOUT_REPORT_IDENTITY')
    termination = summary.get('training_termination')
    metric = 'MSE' if fixed else 'BCE'
    text = ['# 创新 1：SKINNY 单对行扫 A 与列扫 A′', '',
            f"保存科学状态：{saved['status']}；原因：{summary.get('reason', 'UNKNOWN')}。", '',
            '配方：SKINNY64/64、8 轮、差分 0x2000。每组一对、组间独立抽密钥。',
            'A（连续行扫）→ A′（连续列扫），同批密文及标签；固定路线：小块与字各占一半。',
            f'一次 CPU FP32 初始化，双臂独立更新。每轮共用固定 DEV，各自按最低 {metric} 选模，'
            '平分取早；两份磁盘 best 严格重载后才生成共同 TEST。',
            '这不是历史 D 对照、严格二维网格、四密码联合训练或可训练路由的证据。',
            '科学封存不等于命令成功；原生退出以独立监督为准。']
    if fixed:
        text.append('只在固定分层 TRAIN/DEV 完整生成并保存后才更新；每轮只重排 TRAIN 样本。'
                    '训练目标和两版早停均使用 sigmoid 概率的 MSE。')
        text.append('与上次在线生成的实验相比，固定数据、批大小和损失函数同时改变；'
                    '不能把总体涨跌归因于其中任何一项。')
    if termination is not None:
        text.append(f"训练循环结束：{termination['reason']}；"
                    f"完整轮数：{termination['completed_epochs']}。")
    if summary.get('synthetic'):
        text.append('此记录使用人工输入／替身，不是科学训练结果。')
    if saved['status'] == 'SUCCEEDED':
        paired = summary['paired_test']
        lo, hi = paired['ci95']
        mse_column = 'MSE|' if fixed else ''
        mse_alignment = '---:|' if fixed else ''
        text.extend(['', '## 保存的 TEST 成绩', '',
                     f"恒常预测参考 BCE ln(2) = {summary['constant_bce']:.8f}。", '',
                     f'|组别|最佳轮|{mse_column}BCE|BCE − ln(2)|ACC|',
                     f'|---|---:|{mse_alignment}---:|---:|---:|'])
        for slot, label in zip(SLOTS, ('A（行扫）', 'A′（列扫）'), strict=True):
            score = summary['test_scores'][slot]
            mse_value = f"{score['mse']:.8f}|" if fixed else ''
            text.append(f"|{label}|{summary['best'][slot]['epoch']}|{mse_value}{score['bce']:.8f}|"
                        f"{summary['bce_minus_constant'][slot]:+.8f}|{score['accuracy']:.4%}|")
        direction = {
            'lower': '区间全负，本次数据支持列扫 BCE 较低。',
            'higher': '区间全正，本次数据支持列扫 BCE 退化。',
            'uncertain': '区间跨零或触零，方向不确定；不等于没有影响。',
        }[paired['direction']]
        text.extend(['', f"同组 ΔBCE = BCE(A′) − BCE(A) = {paired['delta_bce']:+.8f}；"
                     f"分层标准误 {paired['standard_error']:.8f}，"
                     f'95% 正态近似区间 [{lo:+.8f}, {hi:+.8f}]。{direction}',
                     f"同组 ΔACC（A′ − A）：{paired['delta_acc_pp']:+.4f} 个百分点。",
                     '区间只反映固定两份模型下的 TEST 抽样，不含初始化波动。'
                     '即使区间全负，差异微小或两臂都接近随机也不代表实用提分；'
                     '同批单次对照不是独立复现。'])
    else:
        text.extend(['', '未成功封存，不报告 TEST 成绩。'])
    evidence = dict(run_ref=saved['run_ref'], run_sha256=saved['run_sha256'],
                    updates=summary.get('updates'),
                    paired_completed=summary.get('paired_completed'),
                    counts=saved['counts'], best=summary.get('best'), artifacts=saved['artifacts'])
    if fixed:
        evidence.update(pool_status=summary.get('pool_status'),
                        train_panel=summary.get('train_panel'),
                        dev_panel=summary.get('dev_panel'), test_panel=summary.get('test_panel'))
    text.extend(['', '## 保存证据', '', '```json',
                 json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), '```', ''])
    markdown = '\n'.join(text)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>SKINNY 单对扫描方向对照</title><pre>' + html.escape(markdown) + '</pre></html>')
    source = {k: saved[k] for k in ('run_id', 'run_ref', 'run_sha256')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(dict(schema=1, **source), sort_keys=True).encode()}
