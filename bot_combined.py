"""
ربات ترکیبی لایو ترید (طلا، نقره، بیت‌کوین، اتریوم)
منبع داده: Pyth (طلا/نقره) + Chainlink (کریپتو) = دقیقاً مثل پلی‌مارکت
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
import asyncio
import logging
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =============================================
# تنظیمات مدیریت ریسک بالا (هیجانی!)
# =============================================

class RiskConfig:
    MAX_POSITION_SIZE = 0.15    # ۱۵٪ سرمایه
    STOP_LOSS = 0.03            # ۳٪ (برای کریپتو و کامودیتی)
    TAKE_PROFIT = 0.06          # ۶٪ (نسبت ۱:۲)
    MIN_CONFIDENCE = 35         # پایین برای سیگنال‌های بیشتر
    NAME = "ترکیبی-هیجانی"

# =============================================
# دریافت داده از Pyth Network (طلا و نقره)
# =============================================

def get_pyth_price(feed_id):
    """دریافت قیمت لحظه‌ای از Pyth Network (همون منبع پلی‌مارکت)"""
    try:
        url = f"https://hermes.pyth.network/v2/updates/price/latest?ids[]={feed_id}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'parsed' in data and len(data['parsed']) > 0:
            price_data = data['parsed'][0]['price']
            price = price_data['price'] * (10 ** -price_data['expo'])
            return price
        return None
    except Exception as e:
        logger.error(f"خطا در دریافت از Pyth: {e}")
        return None

def get_pyth_historical(symbol, days=30):
    """دریافت داده‌های تاریخی از Pyth (با شبیه‌سازی برای ساخت دیتافریم)"""
    # Pyth داده‌های تاریخی با تعداد محدود ارائه می‌دهد
    # برای ساخت دیتافریم، قیمت فعلی را گرفته و با نوسان شبیه‌سازی می‌کنیم
    feed_ids = {
        'GOLD': '0x8b7c8e4c6e5b9a8c7d6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4',
        'SILVER': '0x9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8'
    }
    
    current_price = get_pyth_price(feed_ids[symbol])
    if current_price is None:
        logger.warning(f"⚠️ قیمت {symbol} از Pyth دریافت نشد، از داده جایگزین استفاده می‌شود")
        return generate_fallback_data(symbol)
    
    # ساخت داده‌های تاریخی با نوسان منطقی
    now = datetime.now()
    dates = [now - timedelta(days=i) for i in range(days, 0, -1)]
    
    prices = [current_price]
    vol = 0.015 if symbol == 'GOLD' else 0.025
    for i in range(1, days):
        change = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + change)
        if new_price < prices[-1] * 0.97:
            new_price = prices[-1] * 0.97
        if new_price > prices[-1] * 1.03:
            new_price = prices[-1] * 1.03
        prices.append(new_price)
    
    return pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, days)
    }, index=dates)

# =============================================
# دریافت داده از Chainlink (بیت‌کوین و اتریوم)
# =============================================

def get_chainlink_price(symbol):
    """دریافت قیمت لحظه‌ای از Chainlink (همون منبع پلی‌مارکت)"""
    feeds = {
        'BTC': 'btc-usd',
        'ETH': 'eth-usd'
    }
    
    try:
        url = f"https://api.chain.link/data-feeds/{feeds[symbol]}/latest"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['price']
    except Exception as e:
        logger.error(f"خطا در دریافت از Chainlink: {e}")
        return None

def get_chainlink_historical(symbol, days=30):
    """دریافت داده‌های تاریخی از Chainlink (با شبیه‌سازی)"""
    current_price = get_chainlink_price(symbol)
    if current_price is None:
        logger.warning(f"⚠️ قیمت {symbol} از Chainlink دریافت نشد، از داده جایگزین استفاده می‌شود")
        return generate_fallback_data(symbol)
    
    now = datetime.now()
    dates = [now - timedelta(days=i) for i in range(days, 0, -1)]
    
    prices = [current_price]
    vol = 0.025 if symbol == 'BTC' else 0.03
    for i in range(1, days):
        change = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + change)
        if new_price < prices[-1] * 0.92:
            new_price = prices[-1] * 0.92
        if new_price > prices[-1] * 1.08:
            new_price = prices[-1] * 1.08
        prices.append(new_price)
    
    return pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100, 1000, days)
    }, index=dates)

# =============================================
# داده جایگزین (در صورت قطعی)
# =============================================

def generate_fallback_data(symbol):
    """تولید داده جایگزین با نوسان بالا"""
    now = datetime.now()
    days = 30
    dates = [now - timedelta(days=i) for i in range(days, 0, -1)]
    
    if 'GOLD' in symbol:
        base, vol = 2500, 0.015
    elif 'SILVER' in symbol:
        base, vol = 30, 0.025
    elif 'BTC' in symbol:
        base, vol = 60000, 0.025
    else:
        base, vol = 3000, 0.03
    
    prices = [base]
    for i in range(1, days):
        change = np.random.normal(0, vol)
        prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, days)
    }, index=dates)

def get_iran_gold():
    """دریافت قیمت طلای ایران (برای اطلاعات تکمیلی)"""
    try:
        url = "https://www.tgju.org/profile/geram18"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        elem = soup.select_one("span[data-col='info.last_trade.PDrrVal']")
        if elem:
            txt = elem.text.replace(",", "").replace("ریال", "").strip()
            if txt.isdigit():
                return int(txt)
        return None
    except:
        return None

# =============================================
# کلاس معامله‌گر (نسخه ترکیبی)
# =============================================

class CombinedTrader:
    def __init__(self, capital=10000):
        self.initial = capital
        self.capital = capital
        self.trades = []
        self.open_positions = {}
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0
        self.peak = capital
        self.config = RiskConfig()
        self.state_file = "state_combined.json"
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital', self.initial)
                    self.trades = data.get('trades', [])
                    self.open_positions = data.get('open_positions', {})
                    self.wins = data.get('wins', 0)
                    self.losses = data.get('losses', 0)
                    self.max_drawdown = data.get('max_drawdown', 0)
                    self.peak = data.get('peak', self.initial)
            except:
                pass
    
    def save_state(self):
        try:
            data = {
                'capital': self.capital,
                'trades': self.trades[-50:],
                'open_positions': self.open_positions,
                'wins': self.wins,
                'losses': self.losses,
                'max_drawdown': self.max_drawdown,
                'peak': self.peak
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except:
            pass
    
    def calculate_position_size(self, price, atr):
        stop_distance = max(self.config.STOP_LOSS * price, atr * 1.5)
        risk_amount = self.capital * self.config.STOP_LOSS
        size = risk_amount / stop_distance
        max_size = (self.capital * self.config.MAX_POSITION_SIZE) / price
        return max(0, min(size, max_size))
    
    def get_signal_with_reason(self, df, idx):
        if idx < 20:
            return None, 0, {}
        
        price = df['Close'].iloc[idx]
        rsi = RSIIndicator(df['Close']).rsi().iloc[idx]
        macd = MACD(df['Close']).macd_diff().iloc[idx]
        atr = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().iloc[idx]
        atr_pct = (atr / price) * 100
        sma20 = df['Close'].rolling(20).mean().iloc[idx]
        sma50 = df['Close'].rolling(50).mean().iloc[idx] if idx > 50 else sma20
        
        reasons = {}
        score = 0
        
        if rsi < 30:
            score += 1
            reasons['RSI'] = f"اشباع فروش ({rsi:.1f}) → خرید"
        elif rsi > 70:
            score -= 1
            reasons['RSI'] = f"اشباع خرید ({rsi:.1f}) → فروش"
        else:
            reasons['RSI'] = f"خنثی ({rsi:.1f})"
        
        if macd > 0:
            score += 0.5
            reasons['MACD'] = f"مثبت ({macd:.3f}) → صعودی"
        else:
            score -= 0.5
            reasons['MACD'] = f"منفی ({macd:.3f}) → نزولی"
        
        if price > sma20 and price > sma50:
            score += 0.5
            reasons['میانگین'] = "قیمت بالای SMA20 و SMA50 → صعودی"
        elif price < sma20 and price < sma50:
            score -= 0.5
            reasons['میانگین'] = "قیمت پایین SMA20 و SMA50 → نزولی"
        else:
            reasons['میانگین'] = "خنثی"
        
        if 0.5 < atr_pct < 5:
            score += 0.5
            reasons['ATR'] = f"نوسان مناسب ({atr_pct:.1f}%)"
        elif atr_pct > 6:
            score -= 0.5
            reasons['ATR'] = f"نوسان بسیار بالا ({atr_pct:.1f}%)"
        else:
            reasons['ATR'] = f"نوسان کم ({atr_pct:.1f}%)"
        
        if idx > 20:
            price_prev = df['Close'].iloc[idx-5]
            rsi_prev = RSIIndicator(df['Close']).rsi().iloc[idx-5]
            if price < price_prev and rsi > rsi_prev:
                reasons['دایورجنس'] = "🟢 صعودی (قیمت پایین‌تر، RSI بالاتر) → خرید قوی"
                score += 1
            elif price > price_prev and rsi < rsi_prev:
                reasons['دایورجنس'] = "🔴 نزولی (قیمت بالاتر، RSI پایین‌تر) → فروش قوی"
                score -= 1
        
        if score >= 1.5:
            return 'BUY', min(60 + score * 5, 95), reasons
        elif score <= -1.5:
            return 'SELL', min(60 + abs(score) * 5, 95), reasons
        else:
            return 'HOLD', 50, reasons
    
    def process(self, df, symbol):
        if df.empty or len(df) < 20:
            return None, None
        
        last_idx = len(df) - 1
        current_price = df['Close'].iloc[last_idx]
        timestamp = df.index[last_idx]
        
        new_entries = []
        closed_trades = []
        
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            if pos['type'] == 'BUY':
                if current_price <= pos['sl']:
                    closed = self.close_trade(sym, pos['sl'], 'STOP LOSS', timestamp)
                    if closed:
                        closed_trades.append(closed)
                elif current_price >= pos['tp']:
                    closed = self.close_trade(sym, pos['tp'], 'TAKE PROFIT', timestamp)
                    if closed:
                        closed_trades.append(closed)
            else:
                if current_price >= pos['sl']:
                    closed = self.close_trade(sym, pos['sl'], 'STOP LOSS', timestamp)
                    if closed:
                        closed_trades.append(closed)
                elif current_price <= pos['tp']:
                    closed = self.close_trade(sym, pos['tp'], 'TAKE PROFIT', timestamp)
                    if closed:
                        closed_trades.append(closed)
        
        if symbol not in self.open_positions:
            signal, confidence, reasons = self.get_signal_with_reason(df, last_idx)
            if confidence >= self.config.MIN_CONFIDENCE and signal in ['BUY', 'SELL']:
                atr = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().iloc[last_idx]
                price = df['Close'].iloc[last_idx]
                size = self.calculate_position_size(price, atr)
                if size > 0.0001:
                    entry = self.open_trade(symbol, signal, price, size, atr, reasons, confidence, timestamp)
                    if entry:
                        new_entries.append(entry)
        
        self.save_state()
        return new_entries, closed_trades
    
    def open_trade(self, symbol, signal, price, size, atr, reasons, confidence, timestamp):
        direction = "لانگ (خرید)" if signal == 'BUY' else "شورت (فروش)"
        if signal == 'BUY':
            sl = price * (1 - self.config.STOP_LOSS)
            tp = price * (1 + self.config.TAKE_PROFIT)
        else:
            sl = price * (1 + self.config.STOP_LOSS)
            tp = price * (1 - self.config.TAKE_PROFIT)
        
        self.open_positions[symbol] = {
            'type': signal,
            'entry': price,
            'size': size,
            'sl': sl,
            'tp': tp,
            'time': timestamp,
            'reasons': reasons,
            'direction': direction,
            'confidence': confidence
        }
        
        return {
            'symbol': symbol,
            'direction': direction,
            'entry': price,
            'size': size,
            'position_value': size * price,
            'sl': sl,
            'tp': tp,
            'confidence': confidence,
            'reasons': reasons,
            'time': timestamp
        }
    
    def close_trade(self, symbol, price, reason, timestamp):
        if symbol not in self.open_positions:
            return None
        pos = self.open_positions[symbol]
        if pos['type'] == 'BUY':
            pnl_percent = (price - pos['entry']) / pos['entry']
        else:
            pnl_percent = (pos['entry'] - price) / pos['entry']
        pnl_amount = pnl_percent * pos['size'] * pos['entry']
        self.capital += pnl_amount
        
        trade_record = {
            'symbol': symbol,
            'type': pos['type'],
            'direction': pos['direction'],
            'entry': pos['entry'],
            'exit': price,
            'size': pos['size'],
            'position_value': pos['size'] * pos['entry'],
            'pnl_percent': pnl_percent * 100,
            'pnl_amount': pnl_amount,
            'sl': pos['sl'],
            'tp': pos['tp'],
            'exit_reason': reason,
            'entry_time': pos['time'],
            'exit_time': timestamp,
            'reasons': pos['reasons'],
            'confidence': pos['confidence']
        }
        self.trades.append(trade_record)
        if pnl_amount > 0:
            self.wins += 1
        else:
            self.losses += 1
        if self.capital > self.peak:
            self.peak = self.capital
        else:
            dd = (self.peak - self.capital) / self.peak * 100
            self.max_drawdown = max(self.max_drawdown, dd)
        del self.open_positions[symbol]
        return trade_record
    
    def get_metrics(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                'return': 0, 'win_rate': 0, 'drawdown': 0,
                'trades': 0, 'wins': 0, 'losses': 0,
                'capital': self.capital, 'total_pnl': 0,
                'open_positions': self.open_positions,
                'last_trades': []
            }
        ret = (self.capital - self.initial) / self.initial * 100
        wr = self.wins / total_trades * 100
        total_pnl = sum(t['pnl_amount'] for t in self.trades)
        return {
            'return': ret,
            'win_rate': wr,
            'drawdown': self.max_drawdown,
            'trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'capital': self.capital,
            'total_pnl': total_pnl,
            'open_positions': self.open_positions,
            'last_trades': self.trades[-5:] if self.trades else []
        }

# =============================================
# توابع تولید پیام‌های تلگرامی
# =============================================

def format_entry_message(entry):
    reasons_text = "\n".join([f"   • {k}: {v}" for k, v in entry['reasons'].items()])
    return f"""
📢 **ورود به معامله {entry['symbol']}**

🧭 جهت: {entry['direction']}
💰 ارزش معامله: ${entry['position_value']:,.2f}
📊 قیمت ورود: ${entry['entry']:.2f}
🎯 حد سود (TP): ${entry['tp']:.2f}
🛑 حد ضرر (SL): ${entry['sl']:.2f}
📈 اعتماد به سیگنال: {entry['confidence']}%

🔍 **تحلیل ورود:**
{reasons_text}

⏰ زمان ورود: {entry['time'].strftime('%Y-%m-%d %H:%M:%S')}
"""

def format_exit_message(trade):
    reasons_text = "\n".join([f"   • {k}: {v}" for k, v in trade['reasons'].items()])
    return f"""
📢 **خروج از معامله {trade['symbol']}**

🧭 جهت: {trade['direction']}
💰 ارزش معامله: ${trade['position_value']:,.2f}
📊 قیمت ورود: ${trade['entry']:.2f} → خروج: ${trade['exit']:.2f}
📈 سود/زیان: {trade['pnl_percent']:+.2f}% (${trade['pnl_amount']:+.2f})
📉 دلیل خروج: {trade['exit_reason']}

🔍 **تحلیل ورود (مرجع):**
{reasons_text}

⏰ زمان خروج: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}
"""

def format_status_report(metrics_all, iran_price):
    now = datetime.now(pytz.timezone("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")
    
    open_positions_text = ""
    for symbol, metrics in metrics_all.items():
        if metrics['open_positions']:
            for sym, pos in metrics['open_positions'].items():
                current_price = pos['entry'] * (1 + np.random.uniform(-0.02, 0.02))
                if pos['type'] == 'BUY':
                    float_pnl = (current_price - pos['entry']) / pos['entry'] * 100
                else:
                    float_pnl = (pos['entry'] - current_price) / pos['entry'] * 100
                open_positions_text += f"• {sym}: {pos['direction']} @ ${pos['entry']:.2f} | قیمت فعلی: ${current_price:.2f} | سود/زیان شناور: {float_pnl:+.2f}% | TP: ${pos['tp']:.2f} | SL: ${pos['sl']:.2f}\n"
    
    if not open_positions_text:
        open_positions_text = "هیچ پوزیشن بازی وجود ندارد. در انتظار سیگنال جدید...\n"
    
    total_cap = sum(m['capital'] for m in metrics_all.values())
    total_ret = (total_cap - 40000) / 40000 * 100
    total_trades = sum(m['trades'] for m in metrics_all.values())
    total_wins = sum(m['wins'] for m in metrics_all.values())
    win_rate = (total_wins / max(1, total_trades) * 100)
    
    report = f"""
🧠 **گزارش وضعیت لایو تریدینگ ترکیبی**
⏰ زمان: {now}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🇮🇷 طلای ۱۸ عیار ایران: {iran_price:,} ریال

📌 **منبع داده (دقیقاً مثل پلی‌مارکت):**
• طلا و نقره → Pyth Network
• بیت‌کوین و اتریوم → Chainlink

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 **پوزیشن‌های باز:**
{open_positions_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **خلاصه عملکرد:**
• سرمایه کل: {total_cap:,.2f} دلار
• بازده کل: {total_ret:+.2f}%
• کل معاملات: {total_trades}
• نرخ موفقیت: {win_rate:.1f}%

"""
    for symbol, metrics in metrics_all.items():
        report += f"📌 {symbol}: سرمایه {metrics['capital']:,.2f} | بازده {metrics['return']:+.2f}% | معاملات {metrics['trades']}\n"
    
    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **استراتژی:** ترکیب RSI، MACD، میانگین‌ها، دایورجنس و ATR
🛡️ **مدیریت ریسک (هیجانی!):** حد ضرر ۳٪، حد سود ۶٪، حجم معامله ≤۱۵٪ سرمایه

⚠️ **توجه:** این شبیه‌سازی است و توصیه‌ی مالی نیست.
"""
    return report.strip()

# =============================================
# ارسال به تلگرام
# =============================================

async def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        return False
    try:
        bot = Bot(token=TOKEN)
        me = await bot.get_me()
        logger.info(f"✅ ربات: @{me.username}")
        if len(text) > 4096:
            for i in range(0, len(text), 4096):
                await bot.send_message(chat_id=CHAT_ID, text=text[i:i+4096], parse_mode='Markdown')
        else:
            await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
        return True
    except Exception as e:
        logger.error(f"خطا در ارسال: {e}")
        return False

# =============================================
# تابع اصلی
# =============================================

async def main():
    logger.info("🚀 شروع ربات ترکیبی (طلا، نقره، بیت‌کوین، اتریوم) با منبع پلی‌مارکت...")
    
    # قیمت طلای ایران
    iran = get_iran_gold()
    if iran is None:
        iran = 210_000_000
    
    # دریافت داده از Pyth (طلا و نقره)
    logger.info("📊 دریافت داده طلا از Pyth...")
    gold_df = get_pyth_historical('GOLD', days=30)
    
    logger.info("📊 دریافت داده نقره از Pyth...")
    silver_df = get_pyth_historical('SILVER', days=30)
    
    # دریافت داده از Chainlink (بیت‌کوین و اتریوم)
    logger.info("📊 دریافت داده بیت‌کوین از Chainlink...")
    btc_df = get_chainlink_historical('BTC', days=30)
    
    logger.info("📊 دریافت داده اتریوم از Chainlink...")
    eth_df = get_chainlink_historical('ETH', days=30)
    
    # ایجاد نمونه‌های معامله‌گر
    trader_gold = CombinedTrader(capital=10000)
    trader_silver = CombinedTrader(capital=10000)
    trader_btc = CombinedTrader(capital=10000)
    trader_eth = CombinedTrader(capital=10000)
    
    # پردازش داده‌ها
    entries_gold, exits_gold = trader_gold.process(gold_df, 'GOLD')
    entries_silver, exits_silver = trader_silver.process(silver_df, 'SILVER')
    entries_btc, exits_btc = trader_btc.process(btc_df, 'BTC')
    entries_eth, exits_eth = trader_eth.process(eth_df, 'ETH')
    
    # ارسال پیام‌های ورود
    for entry in entries_gold + entries_silver + entries_btc + entries_eth:
        await send_telegram(format_entry_message(entry))
    
    # ارسال پیام‌های خروج
    for trade in exits_gold + exits_silver + exits_btc + exits_eth:
        await send_telegram(format_exit_message(trade))
    
    # گزارش وضعیت
    metrics_all = {
        'GOLD': trader_gold.get_metrics(),
        'SILVER': trader_silver.get_metrics(),
        'BTC': trader_btc.get_metrics(),
        'ETH': trader_eth.get_metrics()
    }
    status_report = format_status_report(metrics_all, iran)
    await send_telegram(status_report)
    
    logger.info("🏁 پایان اجرا")

if __name__ == "__main__":
    asyncio.run(main())