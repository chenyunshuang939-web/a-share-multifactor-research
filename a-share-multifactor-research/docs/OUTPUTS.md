# 输出表速查

| 文件 | 用途 | 关键注意 |
|---|---|---|
| data_quality.csv | 每只股票完整日历、缺行、停牌、无效报价与开盘受限计数 | unknown_missing 不等于确认停牌 |
| label_audit.csv | 两种未来收益及成熟日期 | 事后研究标签，不是信号输入 |
| factor_ic_daily.csv | 主 open-open 标签的原始 IC | date 为因子观测日 |
| factor_ic_close_daily.csv | 辅助 close-close 原始 IC | 仅研究对照 |
| available_ic_daily.csv | 当日收盘已经可用的历史 IC | 动态权重的合法输入 |
| factor_ic_summary.csv | IC 均值、描述性 ICIR、HAC t/p | 未做多重检验修正 |
| dynamic_factor_weights.csv | 因子权重，绝对值和为 1 | 缺失为未完成预热/统计不可用，不伪造零信号 |
| scored_panel.csv | 全股票每日评分与 11 个 Z 因子 | 缺因子或不可选股票不评分 |
| latest_signals.csv | 最后观测日全部股票评分排序 | 不保证下一交易日一定能成交，不是买入建议 |
| factor_group_returns.csv | 每日每因子每组 H 期收益与股票数 | 重叠持有期，不直接逐行复利 |
| factor_group_summary.csv | 各组平均 H 期收益与组规模 | 用于单调性分析 |
| factor_group_spread.csv | 最高减最低组收益及 HAC 统计 | 研究价差，不是计费可成交多空净值 |
| factor_rank_corr.csv | 日横截面秩相关矩阵平均 | 不等于时间序列收益相关性 |
| holdings.csv | 每次调仓的信号日、执行日及目标权重 | 目标持仓，不保证全部成交 |
| trades.csv | 实际逐笔买卖、数量、价格、成交额和费用 | 可与每日资金变化对账 |
| positions_daily.csv | 每日开盘执行后的实际份额、估值与权重 | 停牌/受阻仓位仍保留 |
| execution_events.csv | 无信号、买卖受阻等事件 | 有完整 CSV 表头，即使没有事件 |
| portfolio_daily.csv | NAV、收益、现金、股票市值、换手率、费用 | 开盘执行后观察点；末日不强制平仓 |
| static_daily.csv / static_reduced_daily.csv / equal_weight_daily.csv | 三个同账本对照 | 相同基础可选条件和首次信号日 |
| comparison_daily.csv | 对齐后的所有策略与指数 | 统一比较区间 |
| performance_summary.csv | 总收益、CAGR、标准 Sharpe、回撤、跟踪误差等 | rf=0；指数不收交易费 |
| period_performance.csv | 回顾性时间分段表现 | 不是从未见过的 holdout |
| sensitivity.csv | 指定费用、Top N、调仓间隔实验 | 只有 --experiments 时有结果；不自动挑最优 |
| rolling_var.csv | 逐日事前 VaR 与突破标记 | 初期样本不足为缺失 |
| risk_report.csv | 最近窗口 VaR/ES、下一期 VaR 及 Kupiec | 99% ES 尾部样本很少 |
| worst_5_days.csv | 历史最差五个观察日 | 不是情景压力测试 |
| run_manifest.json | 参数、版本、原始/处理数据和源码校验和 | 复现和结果归属依据 |
| equity_curve.png / cumulative_rank_ic.png | 比较净值和原始 IC 累积图 | 累积 IC 不是交易利润 |
