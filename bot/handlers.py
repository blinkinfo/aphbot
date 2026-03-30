"""Telegram command and callback-query handlers."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

import openpyxl
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config as cfg
from bot.formatters import (
    format_help,
    format_recent_signals,
    format_recent_trades,
    format_signal_stats,
    format_status,
    format_trade_stats,
)
from bot.keyboards import (
    back_to_menu,
    download_keyboard,
    main_menu,
    settings_keyboard,
    signal_filter_row,
    trade_filter_row,
)
from bot.middleware import auth_check
from db import queries
from polymarket import account as pm_account

log = logging.getLogger(__name__)

# Set at startup by main.py
_start_time: datetime = datetime.now(timezone.utc)
_poly_client: Any = None


def set_poly_client(client: Any) -> None:
    global _poly_client
    _poly_client = client


def set_start_time() -> None:
    global _start_time
    _start_time = datetime.now(timezone.utc)


def _uptime() -> str:
    delta = datetime.now(timezone.utc) - _start_time
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


# ---------------------------------------------------------------------------
# Safe edit helper — silently ignores 'Message is not modified' errors
# ---------------------------------------------------------------------------

async def _safe_edit(query, text, reply_markup=None, parse_mode="HTML"):
    """Edit a message, silently ignoring 'not modified' errors."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            pass  # Content unchanged — not an error
        else:
            raise


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@auth_check
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "\U0001f916 <b>Welcome to AutoPoly!</b>\n\n"
        "BTC Up/Down 5-min trading bot for Polymarket.\n"
        "Select an option below:"
    )
    await update.message.reply_text(text, reply_markup=main_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@auth_check
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    connected = False
    balance = None
    positions = []
    if _poly_client:
        connected = await pm_account.get_connection_status(_poly_client)
        balance = await pm_account.get_balance(_poly_client)
        positions = await pm_account.get_open_positions(_poly_client)

    autotrade = await queries.is_autotrade_enabled()
    trade_amount = await queries.get_trade_amount()
    last_sig = await queries.get_last_signal()
    last_sig_str = None
    if last_sig:
        ss = last_sig["slot_start"].split(" ")[-1] if " " in last_sig["slot_start"] else last_sig["slot_start"]
        last_sig_str = f"{ss} UTC ({last_sig['side']})"

    text = format_status(
        connected=connected,
        balance=balance,
        autotrade=autotrade,
        trade_amount=trade_amount,
        open_positions=len(positions),
        uptime_str=_uptime(),
        last_signal=last_sig_str,
    )
    target = update.message if update.message else (update.callback_query.message if update.callback_query else None)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        if target is None:
            return
        await target.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /signals
# ---------------------------------------------------------------------------

async def _render_signals(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_signal_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_signal_stats(stats, label)
    recent = await queries.get_recent_signals(10)
    text += format_recent_signals(recent)
    kb = signal_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_signals(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /trades
# ---------------------------------------------------------------------------

async def _render_trades(update: Update, limit: int | None, active: str) -> None:
    stats = await queries.get_trade_stats(limit=limit)
    label = {"10": "Last 10", "50": "Last 50", "all": "All Time"}[active]
    text = format_trade_stats(stats, label)
    recent = await queries.get_recent_trades(10)
    text += format_recent_trades(recent)
    kb = trade_filter_row(active)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


@auth_check
async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_trades(update, limit=None, active="all")


# ---------------------------------------------------------------------------
# /settings
# ---------------------------------------------------------------------------

@auth_check
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    autotrade = await queries.is_autotrade_enabled()
    trade_amount = await queries.get_trade_amount()
    text = "\u2699\ufe0f <b>Settings</b>\n\nTap a button to change:"
    kb = settings_keyboard(autotrade, trade_amount)
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

@auth_check
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = format_help()
    if update.callback_query:
        await update.callback_query.answer()
        await _safe_edit(update.callback_query, text, reply_markup=back_to_menu())
    else:
        await update.message.reply_text(text, reply_markup=back_to_menu(), parse_mode="HTML")


# ---------------------------------------------------------------------------
# Download handlers
# ---------------------------------------------------------------------------

@auth_check
async def cmd_download_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer("Preparing CSV...")
    rows = await queries.get_all_signals_for_export()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "slot_start", "side", "entry_price", "is_win"])
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    await query.message.reply_document(
        document=io.BytesIO(buf.getvalue().encode()),
        filename="signals.csv",
        caption="\U0001f4e5 All signals export (CSV)",
    )


@auth_check
async def cmd_download_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer("Preparing Excel...")
    rows = await queries.get_all_signals_for_export()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Signals"
    ws.append(["id", "slot_start", "side", "entry_price", "is_win"])
    for r in rows:
        ws.append([r["id"], r["slot_start"], r["side"], r["entry_price"], r["is_win"]])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    await query.message.reply_document(
        document=buf,
        filename="signals.xlsx",
        caption="\U0001f4e5 All signals export (Excel)",
    )


# ---------------------------------------------------------------------------
# Callback query router
# ---------------------------------------------------------------------------

@auth_check
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data == "cmd_menu":
        await query.answer()
        text = "\U0001f916 <b>AutoPoly Menu</b>\n\nSelect an option:"
        await _safe_edit(query, text, reply_markup=main_menu())

    elif data == "cmd_status":
        await cmd_status(update, context)

    elif data == "cmd_signals":
        await _render_signals(update, limit=None, active="all")

    elif data == "cmd_trades":
        await _render_trades(update, limit=None, active="all")

    elif data == "cmd_settings":
        await cmd_settings(update, context)

    elif data == "cmd_help":
        await cmd_help(update, context)

    # Signal filters
    elif data == "signals_10":
        await _render_signals(update, limit=10, active="10")
    elif data == "signals_50":
        await _render_signals(update, limit=50, active="50")
    elif data == "signals_all":
        await _render_signals(update, limit=None, active="all")

    # Trade filters
    elif data == "trades_10":
        await _render_trades(update, limit=10, active="10")
    elif data == "trades_50":
        await _render_trades(update, limit=50, active="50")
    elif data == "trades_all":
        await _render_trades(update, limit=None, active="all")

    # Settings
    elif data == "toggle_autotrade":
        current = await queries.is_autotrade_enabled()
        await queries.set_setting("autotrade_enabled", "false" if current else "true")
        await cmd_settings(update, context)

    elif data == "change_amount":
        await query.answer()
        await _safe_edit(
            query,
            "\U0001f4b5 <b>Set Trade Amount</b>\n\n"
            "Type the new amount in USDC (e.g. <code>2.50</code>):",
        )
        context.user_data["awaiting_amount"] = True

    elif data == "download_csv":
        await cmd_download_csv(update, context)

    elif data == "download_xlsx":
        await cmd_download_excel(update, context)

    else:
        await query.answer("Unknown action")


# ---------------------------------------------------------------------------
# Text handler (for trade amount input)
# ---------------------------------------------------------------------------

@auth_check
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_amount"):
        return

    context.user_data["awaiting_amount"] = False
    raw = update.message.text.strip().replace("$", "")
    try:
        amount = float(raw)
        if amount <= 0:
            raise ValueError("non-positive")
    except ValueError:
        await update.message.reply_text(
            "\u274c Invalid amount. Please enter a positive number (e.g. 2.50)."
        )
        return

    amount = round(amount, 2)
    await queries.set_setting("trade_amount_usdc", str(amount))
    await update.message.reply_text(
        f"\u2705 Trade amount updated to <b>${amount:.2f}</b>",
        parse_mode="HTML",
    )
    # Show settings panel again
    autotrade = await queries.is_autotrade_enabled()
    kb = settings_keyboard(autotrade, amount)
    await update.message.reply_text(
        "\u2699\ufe0f <b>Settings</b>",
        reply_markup=kb,
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Register all handlers
# ---------------------------------------------------------------------------

def register(application) -> None:
    """Attach all command and callback handlers to the Telegram Application."""
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("signals", cmd_signals))
    application.add_handler(CommandHandler("trades", cmd_trades))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    async def _error_handler(update, context):
        log.error("Telegram error: %s", context.error)
        # Don't re-raise — just log it so the bot keeps running

    application.add_error_handler(_error_handler)
