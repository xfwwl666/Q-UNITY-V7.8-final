# Q-UNITY V7.8-final 深度代码审计报告

**审计版本：** Q-UNITY V7.8-final（build_v7.8_final.py 构建）  
**审计日期：** 2026-02-25  
**审计人：** 高级量化系统安全审计师 / 金融软件架构师  
**审计范围：** 全量源代码（8,372 行，含 tests/）

---

## 第一部分：执行摘要

### 一句话结论

> **当前版本不可直接用于实盘交易。** 存在 3 项致命缺陷（T+1 规则缺失、每日亏损限制名存实亡、每日亏损从不重置），以及多项严重合规和风控漏洞，若接入真实券商资金将面临直接损失风险。

### 最关键风险点（Top 5）

| # | 风险 | 严重级别 | 潜在影响 |
|---|------|---------|---------|
| 1 | **SimulatedTrader 未实施 T+1**：当日买入当日即可止损/止盈卖出，违反 A 股强制规则 | 🔴 致命 | 实盘强制撤单、账户冻结 |
| 2 | **每日最大亏损限制配置有值但代码从不检查**：`max_daily_loss_pct` 是"摆设"，无法阻止连续亏损买入 | 🔴 致命 | 单日亏损失控 |
| 3 | **`_daily_loss` 永不重置**：持久化后跨日累积，导致风控失真（或修复后永久冻结买入） | 🔴 致命 | 风控完全失效 |
| 4 | **告警发送同步阻塞监控线程**：email/telegram 在监控主线程调用，SMTP 连接挂起可致扫描停滞 | 🟠 严重 | 实时监控停摆 |
| 5 | **实时策略缓存无容量上限**：监控 5000 只股票时字典无界增长，长期运行内存耗尽 | 🟠 严重 | OOM 崩溃 |

### 系统成熟度评分

```
整体评分：4.5 / 10
  ├─ 回测引擎（BacktestEngine）：7/10  ✅ 架构扎实，NB 系列修复质量良好
  ├─ 实时监控（MonitorEngine）：5/10  ⚠️ 信号合并逻辑正确，但线程安全有隐患
  ├─ 模拟交易（SimulatedTrader）：2/10 ❌ 致命缺陷集中，不可用于实盘参考
  ├─ 数据采集（Pipeline）：6/10      ⚠️ A-04/A-07 修复正确，列名不一致遗留
  └─ 合规性（A 股规则）：3/10       ❌ T+1/涨跌停均未落实
```

---

## 第二部分：修复验证清单（A-01 ~ A-15）

| 编号 | 修复内容 | 结果 | 证据 / 备注 |
|------|---------|------|------------|
| **A-01** | `price_arrays` 列名 `vol → volume` | ✅ 正确 | `menu_main.py:738` `df.rename(columns={"vol": "volume"}, inplace=True)`；`menu_main.py:843` `"volume": df_aligned["volume"]`。⚠️ **注意**：`pipeline.py` 原始存储仍用 `vol`，rename 仅在回测路径；`DataValidator.REQUIRED_COLS` 仍为 `"vol"`，两者存在列名不一致（见新缺陷 NF-04）。 |
| **A-02** | 多策略对比胜率键名 `win_rate → trade_win_rate` | ✅ 正确 | `menu_main.py:986` `fmt_pct(perf.get("trade_win_rate"))`；`menu_main.py:1301` `wr = perf.get("trade_win_rate", 0) or 0` |
| **A-03** | `factor_data` 切片 `searchsorted` 索引类型守卫 | ✅ 正确（有残留风险） | `menu_main.py:874-891` 将索引统一转 `str` 后再 `searchsorted`。⚠️ 异常 fallback 直接返回完整 `fdf`（含未来数据），仍可能引入前视偏差（见新缺陷 NF-09）。 |
| **A-04** | AKShare 延迟参数 `1.5/3.5` 秒 | ✅ 正确 | `pipeline.py:362-363` `ak_delay_min: float = 1.5`，`ak_delay_max: float = 3.5` |
| **A-05** | `ConfigManager` 深拷贝 | ✅ 正确 | `src/config.py:51` `self.config: Dict[str, Any] = copy.deepcopy(self._DEFAULT)` |
| **A-06** | 移除幽灵策略 `sector_momentum` | ✅ 正确 | `menu_main.py:1502,1513` 注释明确标记已移除 |
| **A-07** | Linux `fork` / Windows `spawn` | ✅ 正确 | `pipeline.py:626-629` `_spawn_method = "fork" if _platform.system() == "Linux" else "spawn"` |
| **A-08** | `importlib` 替代 `exec()` | ✅ 正确 | `main.py:18-19,127,133` `importlib.import_module(mod)` |
| **A-09** | 多策略对比 `ProcessPoolExecutor` 并行化 | ✅ 正确 | `menu_main.py:1283+` 多策略回测并行路径（代码中有 ProcessPoolExecutor 实现） |
| **A-10** | 版本号 `V7.1 → V7.8` | ✅ 正确 | `main_op.py:4` 文件标题含 `V7.8`；`main_op.py:148` argparse description |
| **A-11** | 移除 Tushare 心跳 | ✅ 正确 | `menu_main.py:71` 注释标记 `A-11 Fix: 移除 Tushare DNS 心跳检测` |
| **A-12** | TDX `adjustflag` 注释修正 | ✅ 正确 | `pipeline.py:131` `df["adjust"] = "hfq"` 含注释说明 TDX 复权类型需实测确认 |
| **A-13** | `price_arrays` 内存警告 | ✅ 正确 | `menu_main.py:831-832` 估算内存消耗并打印警告 |
| **A-14** | `daily_scores` 扩展为多因子矩阵 | ✅ 正确 | `menu_main.py:784-816` 扩展为 `{code: {factor_name: value}}` 格式 |
| **A-15** | `run_report.py` 标准字符字面量 | ✅ 正确 | `run_report.py:156-162` 使用 `"\n"` / `"\t"` 字面量，注释明确 `A-15 Fix` |

**A-01 ~ A-15 总体：15/15 条均已正确应用**，但 A-01、A-03 存在残留风险，详见第三部分。

---

## 第三部分：新发现缺陷（按优先级排序）

### 🔴 致命缺陷

---

#### NF-01：SimulatedTrader 未实施 A 股 T+1 规则

**问题描述**  
`SimulatedTrader.sell()` 方法中无任何对买入日期的检查。当日（T 日）买入的股票，若触发 `update_prices()` 中的止损/止盈判断，会立即被 `monitor.py` 调用 `trader.sell()`，实现当日买当日卖。在 A 股市场，当日买入的股票必须等到 T+1 日方可卖出，真实券商接口会强制拒绝该委托。  

**代码位置**  
- `src/realtime/trader.py`，`sell()` 方法（第 145-173 行）：无 T+1 检查  
- `Position` dataclass（第 37 行）：`holding_days: int = 0`、`open_time: float` 字段存在但从未用于卖出验证  
- `src/realtime/monitor.py:186-189`：`self.trader.sell(t["code"], t.get("price", 0))` 直接调用

**修复建议**

```python
# src/realtime/trader.py  sell() 方法开头添加：
import datetime as _dt

def sell(self, code: str, price: float, shares=None):
    with self._lock:
        if code not in self.positions:
            return {"ok": False, "reason": "position not found"}
        p = self.positions[code]
        # T+1 校验：买入日期 < 今日（自然日比较即可，夜间运行不影响）
        buy_date = _dt.date.fromtimestamp(p.open_time)
        today = _dt.date.today()
        if buy_date >= today:
            return {"ok": False, "reason": f"T+1 限制: 买入日={buy_date}, 今日={today}"}
        # ... 后续逻辑不变
```

---

#### NF-02：每日最大亏损限制名存实亡——代码从未执行限制

**问题描述**  
`config.json` 配置了 `realtime.trading.max_daily_loss_pct = 0.05`（5%），`RiskParams` 类中也正确保存了 `max_daily_loss_pct`，但 `buy()` 方法中**从未检查** `_daily_loss` 是否超过限额。即使当日已亏损 10% 仍会继续买入，配置项完全无效。

**代码位置**  
- `src/realtime/trader.py:103-142`（`buy()` 方法）：无 `_daily_loss` 检查  
- `src/realtime/trader.py:64`：`max_daily_loss_pct: float = 0.03`（已配置但未使用）

**修复建议**

```python
# buy() 方法开头（获取锁后）添加：
total_assets = self.cash + sum(p.market_value for p in self.positions.values())
daily_loss_limit = total_assets * self.risk.max_daily_loss_pct
if -self._daily_loss >= daily_loss_limit:
    return {"ok": False, "reason": f"当日亏损限额 ({self._daily_loss:.2f}) 已触发"}
```

---

#### NF-03：`_daily_loss` 永不重置——跨日累积导致风控失真

**问题描述**  
`SimulatedTrader._daily_loss` 在 `_save()` / `_load()` 中被持久化，但系统中**没有任何机制**在新交易日开始时将其清零。结果有两种：（a）当前：`max_daily_loss_pct` 未被检查，该字段仅作为统计展示，虚假地累积历史损失；（b）若 NF-02 被修复：由于 `_daily_loss` 永不重置，系统在首次触发每日亏损限制后，将在所有后续交易日永久禁止买入。

**代码位置**  
- `src/realtime/trader.py:165`：`self._daily_loss += min(pnl, 0)`（只增不减）  
- `src/realtime/trader.py:229,242`：`_save()`/`_load()` 包含 `daily_loss`，跨会话持久

**修复建议**  
在 `update_prices()` 开头或 `MonitorEngine` 每日首次扫描时触发重置：

```python
# trader.py 新增方法
def reset_daily_stats(self) -> None:
    """在每个交易日开始时调用（由 MonitorEngine 或调度器负责）"""
    with self._lock:
        self._daily_loss = 0.0
        self._save()
```

同时在 `monitor.py` 的 `scan_once()` 中，当检测到新的交易日时调用 `self.trader.reset_daily_stats()`。

---

### 🟠 严重缺陷

---

#### NF-04：DataValidator 列名与实际 Parquet 列名不一致（`vol` vs `volume`）

**问题描述**  
`pipeline.py` 的 `_clean_tdx_bars()` 将 TDX 原始数据存储为 `vol` 列（第 113-116 行）。`DataValidator.REQUIRED_COLS = {"date", "open", "high", "low", "close", "vol"}`（validator.py:23）与此一致，验证通过。但 A-01 修复仅在 `menu_main.py` 回测路径做了 `rename({"vol": "volume"})`（第 738 行），Parquet 文件中实际存储的仍是 `vol`。问题出在：回测路径读取 parquet 后 rename，而实时监控路径（`monitor.py._get_cached_parquet()`）直接读取 parquet 后传给策略，策略中若引用 `df["volume"]` 会 `KeyError`。

**代码位置**  
- `src/data/collector/validator.py:23`：`REQUIRED_COLS` 含 `"vol"`  
- `src/data/collector/pipeline.py:113-116`：存储为 `vol`  
- `menu_main.py:738`：仅在回测路径 rename  
- `src/realtime/monitor.py:_get_cached_parquet()`：直接透传，未 rename  

**修复建议**  
选择方案 A（推荐）：在 `pipeline.py` 的 `_clean_tdx_bars()` 中直接将列名统一为 `volume`，同步更新 `REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}`，并删除 `menu_main.py:738` 的重复 rename。  
或方案 B：在 `_get_cached_parquet()` 返回前统一做一次 rename。

---

#### NF-05：告警发送同步阻塞监控线程

**问题描述**  
`alerter.py` 的 `send_alert()` 方法在监控主线程（`MonitorEngine._loop()` → `scan_once()`）中**同步调用** `_send_email()`、`_send_telegram()`、`_send_dingtalk()`。其中 email 使用 `smtplib.SMTP_SSL`（无显式连接超时），SMTP 握手失败或网络不稳定时可能阻塞 30 秒以上，导致整个扫描周期延误，错过后续价格更新。

**代码位置**  
- `src/realtime/alerter.py:105-111`（`send_alert()`）：顺序同步调用各渠道  
- `src/realtime/alerter.py:155-165`（`_send_email()`）：`SMTP_SSL` 无连接超时参数  
- `src/realtime/alerter.py:106`：在 `scan_once()` 调用栈内

**修复建议**  
将各渠道发送改为后台线程：

```python
# alerter.py send_alert() 中将同步调用改为：
import threading
if self._email_enabled:
    threading.Thread(target=self._send_email, args=(full_subject, body),
                     daemon=True).start()
# 同理对 dingtalk/telegram/wechat

# _send_email 中增加 SMTP 超时：
smtplib.SMTP_SSL(host, port, timeout=10)
```

---

#### NF-06：实时策略缓存字典无容量上限——OOM 风险

**问题描述**  
各策略的 `generate_realtime_signal()` 使用字典缓存各股票的最新因子值（如 `RSRSMomentumStrategy._last_rsrs: Dict[str, tuple]`），TTL 为 60 秒。但字典无最大容量限制，监控 A 股全市场（约 5000 只）时，每次扫描后字典将持续增长。长期运行中若股票被加入/退出监控但字典不清理，旧条目永不删除，内存稳定泄漏。

**代码位置**  
- `src/strategy/strategies.py:733`：`self._last_rsrs: Dict[str, tuple] = {}`  
- `src/strategy/strategies.py:785`：`self._last_score: Dict[str, tuple] = {}`  
- `src/strategy/strategies.py:826`：`self._last_rsrs_adv: Dict[str, tuple] = {}`

**修复建议**  
替换为 `functools.lru_cache` 或自实现 LRU，或在缓存中增加容量限制：

```python
from collections import OrderedDict

class _BoundedTTLCache:
    def __init__(self, maxsize=6000, ttl=60.0):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key, now):
        if key in self._cache:
            ts, val = self._cache[key]
            if now - ts < self._ttl:
                self._cache.move_to_end(key)
                return val, True
            del self._cache[key]
        return None, False

    def set(self, key, val, now):
        self._cache[key] = (now, val)
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
```

---

#### NF-07：SimulatedTrader 仓位计算基于 `cash` 而非总资产，导致超限

**问题描述**  
`buy()` 中计算可买入金额：`max_cash = self.cash * self.risk.max_position_pct`（第 114 行）。但 `max_position_pct` 的语义应为"单笔仓位不超过**总资产**的 X%"，而非"不超过**现金**的 X%"。当大量资金已建仓后，剩余现金减少，基于 cash 计算会允许后续买入使单一持仓超过总资产 10% 的限制（因为分母偏小）。相反，在初始现金充足时，又可能比预期多买。

**代码位置**  
- `src/realtime/trader.py:114`

**修复建议**

```python
# 替换为：
mv = sum(p.market_value for p in self.positions.values())
total_assets = self.cash + mv
max_cash = min(
    total_assets * self.risk.max_position_pct,
    self.cash  # 不能超过可用现金
)
```

---

#### NF-08：`parallel_factor_precomputation` 强制使用 `spawn`，与 pipeline 不一致

**问题描述**  
`execution.py` 的 `parallel_factor_precomputation()` 硬编码使用 `spawn` 进程上下文（第 `ctx = _mp.get_context("spawn")`），而 `pipeline.py`（A-07 修复后）在 Linux 上使用 `fork`。在同一次回测中若两者同时使用，会出现上下文不一致；在 Linux 上 `spawn` 比 `fork` 慢约 2-5 倍，且需要完整模块序列化，对重度依赖 numpy 的因子计算影响显著。

**代码位置**  
- `src/engine/execution.py`：`parallel_factor_precomputation()` 约第 770 行，`ctx = _mp.get_context("spawn")`

**修复建议**  
与 `pipeline.py` 保持一致，根据平台选择：

```python
import platform as _plt
_spawn_method = "fork" if _plt.system() == "Linux" else "spawn"
ctx = _mp.get_context(_spawn_method)
```

---

### 🟡 中等缺陷

---

#### NF-09：A-03 异常 fallback 直接返回完整 factor_data，可能引入前视偏差

**问题描述**  
`menu_main.py:887-891`，当 `searchsorted` 切片逻辑抛出任意异常时，`except Exception` 将整个 `fdf`（含未来数据）赋值给 `cur_factor_slice[code]`，绕过了前视偏差防护（NB-01）。

**代码位置**  
- `menu_main.py:887-891`

**修复建议**  
在异常时跳过该股票，而非 fallback 到完整数据：

```python
except Exception as e:
    logger.warning("factor_data 切片失败 %s: %s，跳过该股票因子", code, e)
    # 不添加到 cur_factor_slice，后续策略对该股票回退到无因子模式
```

---

#### NF-10：随机滑点可产生负值，买入价低于市价（不符合现实）

**问题描述**  
`SimulatedTrader.buy()` 中：`slip = random.uniform(-0.2, 0.2) * self.SLIPPAGE_RATE`（第 109-110 行）。约 50% 概率 `slip < 0`，导致 `exec_price = price * (1 + slip) < price`，即买入价低于市场报价，这在实际市场中不可能发生（除非是量化做市商）。同理 `sell()` 中约 50% 概率卖出价高于市价。此设计会系统性高估策略收益。

**代码位置**  
- `src/realtime/trader.py:109-110`（buy）；`src/realtime/trader.py:153-154`（sell）

**修复建议**

```python
# buy: 买入价只能 >= 市价（正向滑点）
slip = abs(random.uniform(0, 1)) * self.SLIPPAGE_RATE  # [0, SLIPPAGE_RATE]
exec_price = price * (1 + slip)

# sell: 卖出价只能 <= 市价（负向滑点）
slip = abs(random.uniform(0, 1)) * self.SLIPPAGE_RATE
exec_price = price * (1 - slip)
```

---

#### NF-11：DataValidator 允许 `open=0` 且无日期单调性检查（Layer 1 路径）

**问题描述**  
`DataValidator.validate()` 仅检查 `close >= 0`（允许停牌日），但未检查 `open == 0`（开盘价为 0 意味着数据异常，并非真实停牌）。停牌日正确数据应为前一日收盘价，而非 0。另外 `validate()` 不检查日期排序（仅 `validate_merge_result()` 检查），若新股数据乱序，回测中 `searchsorted` 会返回错误位置。

**修复建议**  
在 Layer 3 中增加：

```python
# 检查 open <= 0（停牌日 open 应延用前日 close，不应为 0）
zero_open = (numeric_df["open"] <= 0).sum()
if zero_open > len(df) * 0.1:  # 超过 10% 行的 open=0 视为脏数据
    return False, f"too_many_zero_open:{zero_open}_rows"
```

---

#### NF-12：回测与实盘风控参数双套独立，回测结果无法反映实盘风险

**问题描述**  
`config.json` 中存在两套完全独立的风控参数：

- 回测：`risk.stop_loss_pct = 0.10`，`risk.take_profit_pct = 0.20`，`risk.trailing_stop_pct = 0.05`  
- 实盘：`realtime.risk.stop_loss_pct = 0.08`，`realtime.risk.take_profit_pct = 0.20`，`realtime.risk.trailing_stop_pct = 0.05`

止损阈值差异（10% vs 8%）会导致回测与实盘的止损触发频率、资金曲线不一致，回测结果无法真实反映实盘行为。

**修复建议**  
设计统一的风控配置节（如 `risk_unified`），由回测引擎和实盘模块共同读取，或在 `ConfigManager` 中实现自动同步。

---

#### NF-13：无涨跌停价格限制处理

**问题描述**  
回测引擎（`execution.py`）和实时交易（`trader.py`）均未对涨跌停价格进行任何处理：

- 回测中，若某日 K 线为涨停开盘，买入信号仍以开盘价成交，但实际因封板无法成交。
- 实时中，若目标股票处于涨停，`trader.buy()` 会以涨停价计算，但真实券商在涨停时委托无法立即成交（需排队等候）。
- 无法在涨停时避免追高，无法在跌停时执行止损。

**修复建议**  
在 `execution.py._execute_signal()` 中增加涨跌停检测：

```python
prev_close = bar.get("prev_close", 0)
if prev_close > 0:
    limit_up = round(prev_close * 1.10, 2)
    limit_down = round(prev_close * 0.90, 2)
    if sig.side == OrderSide.BUY and exec_price >= limit_up:
        logger.warning("涨停无法买入 %s", sig.code)
        return None
    if sig.side == OrderSide.SELL and exec_price <= limit_down:
        logger.warning("跌停无法卖出 %s，记录为流动性不足", sig.code)
        return None
```

---

### 🔵 低优先级缺陷

---

#### NF-14：敏感配置明文存储，未提供环境变量读取机制

**问题描述**  
`config.json` 中含 `email_password`、`telegram_bot_token`、`dingtalk_webhook`、`wechat_work_webhook` 字段，设计为直接存储明文凭据。若配置文件被提交至版本控制（git）或日志中包含配置转储，将造成凭据泄露。`ConfigManager` 也无从环境变量读取机制。

**修复建议**  
在 `ConfigManager._load()` 后增加环境变量覆盖：

```python
import os
_ENV_MAP = {
    ("realtime", "alert", "email_password"): "QUNITY_EMAIL_PASSWORD",
    ("realtime", "alert", "telegram_bot_token"): "QUNITY_TG_TOKEN",
}
for key_path, env_var in _ENV_MAP.items():
    val = os.environ.get(env_var)
    if val:
        d = self.config
        for k in key_path[:-1]: d = d.setdefault(k, {})
        d[key_path[-1]] = val
```

---

#### NF-15：`_send_email()` 无连接超时参数

**问题描述**  
`alerter.py:160`：`smtplib.SMTP_SSL(host, port)` 未指定 `timeout` 参数，使用系统默认（通常为 OS socket 超时，可达 30-120 秒）。在网络不稳定时阻塞监控线程。（与 NF-05 关联，两者需同时修复）

---

#### NF-16：`open_time` 精度问题——跨夜重启后 T+1 判断可能失效

**问题描述**  
`Position.open_time` 使用 `time.time()`（Unix 时间戳），若模拟交易在同一自然日凌晨（如收盘后运行复盘）买入，再在次日凌晨（如早盘前）重启时卖出，虽日期不同但时间戳差值可能误判。建议改用明确日期而非时间戳。

---

#### NF-17：`majority` 规则的分母使用 `len(results)` 而非 `len(self._strategies)`

**问题描述**  
`monitor.py:_merge_signals()` V7.5 修复后使用 `total = len(results)`（实际返回非 hold 信号的策略数）作为分母。注释称此为"正确"做法，但这意味着：若配置了 5 个策略，其中 3 个返回 `hold`，只有 2 个返回 `buy`，则 `buy_ratio = 2/2 = 1.0 > 0.5`，满足 majority，但实际上只有 40% 的策略支持买入。此行为与 "majority" 语义（超过半数已加载策略同意）不符，会过度触发信号。

**修复建议**  
建议提供配置项让用户选择分母语义，或明确文档说明当前行为。

---

## 第四部分：合规性检查表

| A 股规则 | 系统处理情况 | 状态 |
|---------|------------|------|
| **T+1 限制**（当日买入不可当日卖出） | 回测引擎：`BacktestEngine` 通过 `_pending_signals` 实现 T+1（✅）；实时 `SimulatedTrader`：**完全缺失**（❌） | 🔴 部分不合规 |
| **最小交易单位 100 股** | `execution.py:_calc_buy_budget()` 正确取 100 的倍数（✅）；`trader.py:buy()` 正确取整（✅） | ✅ 合规 |
| **停牌处理** | 回测：`PositionManager.get_total_market_value()` 支持 98% 折价估值（NB-20 ✅）；执行层跳过停牌信号（✅）；实时：`update_prices()` 对停牌股无特殊处理（⚠️） | ⚠️ 部分合规 |
| **涨跌停限制** | 回测和实时均未实现（❌） | 🔴 不合规 |
| **手续费模型**（双边佣金 + 最低 5 元） | 回测：`OrderManager.compute_commission()` 正确（✅）；实时：`max(cost * 0.0003, 5.0)`（✅） | ✅ 合规 |
| **卖出印花税 0.1%** | 回测：`compute_tax()` 仅卖出收取（✅）；实时：`stamp = proceeds * STAMP_TAX`（✅） | ✅ 合规 |
| **滑点方向**（买涨卖跌） | 回测：NB-07 已修复（✅）；实时：方向正确但幅度可为负（⚠️ NF-10） | ⚠️ 部分合规 |
| **除权除息（复权）** | TDX 采用 hfq 后复权（A-12 注释说明需实测，⚠️）；AKShare/BaoStock 明确 hfq（✅） | ⚠️ 待验证 |
| **前视偏差防护**（NB-01） | 回测：T+1 执行信号（✅）；A-03 修复但 fallback 有残留风险（⚠️ NF-09） | ⚠️ 大部分合规 |
| **新股防御**（NB-21 / 5 日新股全 NaN） | `AlphaEngine` 中实现 `_nb21_valid_mask`，测试 T5 验证通过（✅） | ✅ 合规 |
| **熔断冷却**（NB-12） | `BacktestEngine._check_circuit_breaker()` 正确实现（✅）；实时无对应机制（⚠️） | ⚠️ 仅回测有 |
| **每日亏损限制** | 配置和类定义有，代码从不执行（❌ NF-02） | 🔴 不合规 |

---

## 第五部分：结论与路线图

### 最终结论

**系统当前版本不可直接用于实盘交易。**

A-01 ~ A-15 的 15 项历史缺陷均已正确修复，回测引擎（`BacktestEngine`）的核心逻辑较为健壮。然而，**实时交易模块（`SimulatedTrader`）存在 3 项致命缺陷**，将直接导致违反 A 股 T+1 规则或使风控机制形同虚设。在这些问题修复前，该系统仅适合在纯回测模式下使用。

### 实盘前必须修复清单（按优先级排序）

| 优先级 | 缺陷 | 工作量估计 |
|--------|------|-----------|
| P0 | NF-01：T+1 限制（SimulatedTrader.sell） | 30 分钟 |
| P0 | NF-02：每日亏损限制生效（buy 方法增加检查） | 30 分钟 |
| P0 | NF-03：每日亏损重置机制（新增方法 + 调度） | 1 小时 |
| P1 | NF-04：列名 vol/volume 统一（选择方案 A） | 2 小时 |
| P1 | NF-05：告警异步化（线程池） | 2 小时 |
| P1 | NF-07：仓位计算基于总资产 | 30 分钟 |
| P1 | NF-10：随机滑点方向修正 | 15 分钟 |
| P2 | NF-13：涨跌停检测 | 4 小时 |

### 建议改进路线图

**短期（1-2 周）：消除致命/严重缺陷**
- 修复 NF-01/02/03（T+1 + 每日亏损）
- 统一 vol/volume 列名（NF-04）
- 告警异步化（NF-05）
- 修复仓位计算和滑点方向（NF-07、NF-10）

**中期（1 个月）：提升合规性和稳定性**
- 实现涨跌停价格检测（NF-13）
- 为实时模块增加熔断机制（对标回测的 circuit breaker）
- 统一回测与实盘风控参数（NF-12）
- 策略缓存增加 LRU 容量限制（NF-06）
- 验证 TDX hfq 复权的实际行为（A-12 注释所示）

**长期（3 个月）：真实实盘接入准备**
- 实现 `BrokerAPI` 的真实券商接口（文件中已定义抽象类）
- 增加环境变量凭据管理（NF-14）
- 补充停牌股在实时模块的折价估值
- 压力测试：5000 只股票长时间运行的内存曲线监控
- 增加回测与实盘信号一致性对比测试用例

---

*报告生成时间：2026-02-25 | 审计基础：build_v7.8_final.py（共 242 行）展开后全量源码 8,372 行*
