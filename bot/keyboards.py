"""Inline keyboard layouts for the Telegram bot."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4ca Status", callback_data="cmd_status"),
            InlineKeyboardButton("\U0001f4e1 Signals", callback_data="cmd_signals"),
        ],
        [
            InlineKeyboardButton("\U0001f4b0 Trades", callback_data="cmd_trades"),
            InlineKeyboardButton("\u2699\ufe0f Settings", callback_data="cmd_settings"),
        ],
        [
            InlineKeyboardButton("\u2753 Help", callback_data="cmd_help"),
        ],
    ])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def settings_keyboard(autotrade_on: bool, trade_amount: float) -> InlineKeyboardMarkup:
    at_label = "\U0001f916 AutoTrade: ON" if autotrade_on else "\U0001f916 AutoTrade: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(at_label, callback_data="toggle_autotrade")],
        [InlineKeyboardButton(f"\U0001f4b5 Trade Amount: ${trade_amount:.2f}", callback_data="change_amount")],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Filter rows (Last 10 / Last 50 / All Time)
# ---------------------------------------------------------------------------

def signal_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            ("[Last 10]" if active == "10" else "Last 10"),
            callback_data="signals_10",
        ),
        InlineKeyboardButton(
            ("[Last 50]" if active == "50" else "Last 50"),
            callback_data="signals_50",
        ),
        InlineKeyboardButton(
            ("[All Time]" if active == "all" else "All Time"),
            callback_data="signals_all",
        ),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [
            InlineKeyboardButton("\U0001f4e5 Download CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4e5 Download Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


def trade_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            ("[Last 10]" if active == "10" else "Last 10"),
            callback_data="trades_10",
        ),
        InlineKeyboardButton(
            ("[Last 50]" if active == "50" else "Last 50"),
            callback_data="trades_50",
        ),
        InlineKeyboardButton(
            ("[All Time]" if active == "all" else "All Time"),
            callback_data="trades_all",
        ),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Back button only
# ---------------------------------------------------------------------------

def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Download keyboard (standalone — kept for direct use if needed)
# ---------------------------------------------------------------------------

def download_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4e5 Download CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4e5 Download Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])
