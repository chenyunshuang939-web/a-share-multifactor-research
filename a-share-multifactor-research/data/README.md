# 本地数据

仓库不包含原始行情或处理缓存。

- 合成演示：`python scripts/run_all.py --demo`，使用独立的 `data/demo/` 目录。
- 真实数据：`python scripts/run_all.py --limit 30 --start 2021-01-01 --end 2026-10-06`。
- 已有合法快照：`python scripts/run_all.py --offline --experiments`。

`download_manifest.json` 保存股票名单、查询覆盖范围和 SHA-256；`build_manifest.json` 保存处理后的选择范围和校验和。只读取 manifest 指定股票，不扫描其他缓存 CSV。

reports/ 中的真实统计来自历史 BaoStock 快照：30 股、2021-01-04 至 2026-09-30。样本中补齐的缺行标记为未知缺失，不能仅凭缺行确认停牌。新下载结果可能因成员名单、数据修订或复权变化而不同。

行情数据和第三方服务使用条件不由项目代码许可证覆盖。
