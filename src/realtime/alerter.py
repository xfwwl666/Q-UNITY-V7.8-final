# -*- coding: utf-8 -*-
"""
src/realtime/alerter.py -- Multi-channel alert module (V7.5)
V7.4: supports multi-strategy signals via send_merged_signal_alert
V7.5: adds Telegram bot and WeChat Work (企业微信) webhook channels
      adds logs/realtime.log output
"""

from __future__ import annotations

import logging
import smtplib
import time
import threading
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LEVEL_EMOJI = {
    "info":    "ℹ️",
    "buy":     "🟢",
    "sell":    "🔴",
    "warning": "⚠️",
    "error":   "❌",
    "profit":  "💰",
    "loss":    "🔻",
}


def _setup_realtime_log_handler():
    """Setup shared realtime.log file handler (V7.5)"""
    rt_logger = logging.getLogger("realtime")
    if not any(isinstance(h, logging.FileHandler) for h in rt_logger.handlers):
        try:
            Path("logs").mkdir(exist_ok=True)
            fh = logging.FileHandler("logs/realtime.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            rt_logger.addHandler(fh)
            rt_logger.setLevel(logging.INFO)
        except Exception:
            pass


class Alerter:
    """多渠道预警通知器 (V7.5)"""

    def __init__(self, config=None):
        self.config = config or {}
        alert_cfg = self.config.get("realtime", {}).get("alert", {})

        # Setup alerts log file (existing)
        log_file = alert_cfg.get("log_file", "logs/realtime_alerts.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self._file_logger = logging.getLogger("realtime.alerts")
        if not self._file_logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._file_logger.addHandler(fh)
            self._file_logger.setLevel(logging.DEBUG)

        # V7.5: also setup realtime.log
        _setup_realtime_log_handler()

        # Email
        self._email_enabled = alert_cfg.get("enable_email", False)
        self._email_cfg = alert_cfg

        # DingTalk
        self._dingtalk_enabled = alert_cfg.get("enable_dingtalk", False)
        self._dingtalk_webhook = alert_cfg.get("dingtalk_webhook", "")

        # V7.5: Telegram
        self._telegram_enabled = alert_cfg.get("enable_telegram", False)
        self._telegram_token = alert_cfg.get("telegram_bot_token", "")
        self._telegram_chat_id = alert_cfg.get("telegram_chat_id", "")

        # V7.5: WeChat Work
        self._wechat_enabled = alert_cfg.get("enable_wechat_work", False)
        self._wechat_webhook = alert_cfg.get("wechat_work_webhook", "")

        self._dedup: Dict[str, float] = {}
        self._dedup_window = alert_cfg.get("dedup_window_seconds", 300)
        self._lock = threading.Lock()
        self._handlers: List[Callable[[str, str, str], None]] = []

    def register_handler(self, fn):
        self._handlers.append(fn)

    def send_alert(self, level, subject, body, dedup_key=None):
        key = dedup_key or subject
        now = time.time()
        with self._lock:
            last = self._dedup.get(key, 0)
            if now - last < self._dedup_window:
                return False
            self._dedup[key] = now

        emoji = LEVEL_EMOJI.get(level, "")
        full_subject = emoji + " [" + level.upper() + "] " + subject
        self._file_logger.info("%s | %s", full_subject, body)

        if self._email_enabled:
            self._send_email(full_subject, body)
        if self._dingtalk_enabled and self._dingtalk_webhook:
            self._send_dingtalk(full_subject, body)
        # V7.5: new channels
        if self._telegram_enabled and self._telegram_token:
            self._send_telegram(full_subject, body)
        if self._wechat_enabled and self._wechat_webhook:
            self._send_wechat_work(full_subject, body)

        for fn in self._handlers:
            try:
                fn(level, subject, body)
            except Exception as e:
                logger.warning("Custom alert handler error: %s", e)
        return True

    def send_signal_alert(self, code, name, signal, score, strategy, price, reason=""):
        subject = code + " " + name + " -- " + signal + " signal (score=" + f"{score:.2f}" + ")"
        body = ("strategy: " + strategy
                + " | code: " + code + " | name: " + name + chr(10)
                + "signal: " + signal + " | score: " + f"{score:.4f}"
                + " | price: " + f"{price:.2f}")
        if reason:
            body += chr(10) + "reason: " + reason
        level = "buy" if "buy" in signal.lower() else "sell" if "sell" in signal.lower() else "info"
        return self.send_alert(level, subject, body,
                               dedup_key=code + "_" + signal + "_" + strategy)

    def send_merged_signal_alert(self, code, name, signal, score, price,
                                  triggered_strategies, merge_rule="any"):
        strat_str = ", ".join(triggered_strategies)
        subject = (code + " " + name + " -- [MERGED] " + signal
                   + " (score=" + f"{score:.2f}" + ", rule=" + merge_rule + ")")
        body = ("merged strategies: [" + strat_str + "]" + chr(10)
                + "code: " + code + " | name: " + name + chr(10)
                + "signal: " + signal + " | score: " + f"{score:.4f}"
                + " | price: " + f"{price:.2f}" + chr(10)
                + "merge_rule: " + merge_rule + " | triggered: " + str(len(triggered_strategies)))
        level = "buy" if "buy" in signal.lower() else "sell" if "sell" in signal.lower() else "info"
        return self.send_alert(level, subject, body,
                               dedup_key=code + "_merged_" + signal)

    def send_position_alert(self, code, name, event, pnl_pct, price):
        subject = code + " " + name + " -- " + event + " (pnl=" + f"{pnl_pct:+.1%}" + ")"
        body = ("code: " + code + " | name: " + name + " | event: " + event
                + " | pnl: " + f"{pnl_pct:+.1%}" + " | price: " + f"{price:.2f}")
        level = "profit" if pnl_pct > 0 else "loss"
        return self.send_alert(level, subject, body, dedup_key=code + "_" + event)

    def _send_email(self, subject, body):
        try:
            cfg = self._email_cfg
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = cfg.get("email_from", "")
            msg["To"] = cfg.get("email_to", "")
            with smtplib.SMTP_SSL(cfg.get("email_smtp_host", "smtp.gmail.com"),
                                  int(cfg.get("email_smtp_port", 465))) as srv:
                srv.login(cfg.get("email_user", ""), cfg.get("email_password", ""))
                srv.sendmail(msg["From"], [msg["To"]], msg.as_string())
        except Exception as e:
            logger.warning("Email alert failed: %s", e)

    def _send_dingtalk(self, subject, body):
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "msgtype": "text",
                "text": {"content": subject + chr(10) + body}
            }).encode()
            req = urllib.request.Request(
                self._dingtalk_webhook,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning("DingTalk alert failed: %s", e)

    def _send_telegram(self, subject, body):
        """V7.5: Send alert via Telegram Bot API"""
        try:
            import urllib.request, urllib.parse, json as _json
            text = subject + chr(10) + chr(10) + body
            payload = _json.dumps({
                "chat_id": self._telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
            }).encode("utf-8")
            url = "https://api.telegram.org/bot" + self._telegram_token + "/sendMessage"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.debug("Telegram alert sent to chat_id=%s", self._telegram_chat_id)
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    def _send_wechat_work(self, subject, body):
        """V7.5: Send alert via WeChat Work (企业微信) group robot webhook"""
        try:
            import urllib.request, json as _json
            content = subject + chr(10) + body
            payload = _json.dumps({
                "msgtype": "text",
                "text": {"content": content},
            }).encode("utf-8")
            req = urllib.request.Request(
                self._wechat_webhook,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.debug("WeChat Work alert sent")
        except Exception as e:
            logger.warning("WeChat Work alert failed: %s", e)