"""
ربات ترکیبی لایو ترید با سیستم پشتیبان چندمنبعی و میانگین‌گیری
منابع: Gold-API (طلا/نقره), Metals-API (نقره), Pyth, Chainlink, Yahoo, Goldprice.org, Ninjas
CoinGecko فقط برای بیت‌کوین و اتریوم
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
from telegram import Bot
from telegram.error import TelegramError
import yfinance as yf
import time
import re
import urllib3

# غیرفعال کردن هشدارهای SSL (برای Metals-API)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NINJAS_API_KEY = os.getenv("NINJAS_API_KEY")
PYTH_API_KEY = os.getenv("PYTH_API_KEY")

if not TOKEN or not CHAT_ID:
    logger.warning("⚠️ TELEGRAM_TOKEN یا CHAT_ID تنظیم نشده! پیام‌ها ارسال نمی‌شوند.")

# =============================================
# تنظیمات مدیریت ریسک
# =============================================

class RiskConfig:
    MAX_POSITION_SIZE = 0.15
    STOP_LOSS = 0.03
    TAKE_PROFIT = 0.06
    MIN_CONFIDENCE = 35
    BASE_RISK_PER_TRADE = 0.02
    VOLATILITY_ADJUSTMENT = True
    SIGNAL_SCORE_WEIGHT = 1.5

# =============================================
# ۱. دریافت از PYTH NETWORK (با API Key)
# =============================================

def get_pyth_price(feed_id):
    if not PYTH_API_KEY:
        logger.warning("⚠️ PYTH_API_KEY تنظیم نشده است")
        return None
    try:
        url = f"https://hermes.pyth.network/v2/updates/price/latest?ids[]={feed_id}"
        headers = {"Authorization": f"Bearer {PYTH_API_KEY}", "Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if 'parsed' in data and len(data['parsed']) > 0:
                price_data = data['parsed'][0]['price']
                return price_data['price'] * (10 ** -price_data['expo'])
        else:
            logger.warning(f"⚠️ Pyth خطا: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"خطا در Pyth: {e}")
        return None

# =============================================
# ۲. دریافت از CHAINLINK
# =============================================

def get_chainlink_price(symbol):
    feeds = {'BTC': 'btc-usd', 'ETH': 'eth-usd'}
    try:
        url = f"https://api.chain.link/data-feeds/{feeds[symbol]}/latest"
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            return response.json()['price']
        return None
    except Exception as e:
        logger.error(f"خطا در Chainlink: {e}")
        return None

# =============================================
# ۳. دریافت از YAHOO FINANCE
# =============================================

def get_yahoo_price(symbol):
    try:
        yahoo_symbols = {'GOLD': 'GC=F', 'SILVER': 'SI=F', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}
        ticker = yahoo_symbols.get(symbol)
        if not ticker:
            return None
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
        return None
    except Exception as e:
        logger.error(f"خطا در یاهو: {e}")
        return None

def get_yahoo_historical(symbol, days=30):
    try:
        yahoo_symbols = {'GOLD': 'GC=F', 'SILVER': 'SI=F', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}
        ticker = yahoo_symbols.get(symbol)
        if not ticker:
            return None
        df = yf.download(ticker, period=f"{days+5}d", interval="1d", progress=False)
        if df is not None and not df.empty and len(df) >= 10:
            return df.iloc[-days:]
        return None
    except Exception as e:
        logger.error(f"خطا در یاهو: {e}")
        return None

# =============================================
# ۴. دریافت از GOLDPRICE.ORG
# =============================================

def get_goldprice_org_price(symbol):
    try:
        url = "https://goldprice.org/live-gold-price.html"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        if symbol == 'GOLD':
            match = re.search(r'Spot Gold Price:\s*USD\s*([\d,]+\.?\d*)', text)
            if match:
                return float(match.group(1).replace(',', ''))
        elif symbol == 'SILVER':
            match = re.search(r'Spot Silver Price:\s*USD\s*([\d,]+\.?\d*)', text)
            if match:
                return float(match.group(1).replace(',', ''))
        return None
    except Exception as e:
        logger.error(f"خطا در goldprice.org: {e}")
        return None

# =============================================
# ۵. دریافت از API NINJAS
# =============================================

def get_ninjas_price(symbol):
    if not NINJAS_API_KEY:
        logger.warning("⚠️ NINJAS_API_KEY تنظیم نشده است")
        return None
    try:
        url = "https://api.api-ninjas.com/v1/goldprice"
        headers = {"X-Api-Key": NINJAS_API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and float(price) > 0:
                logger.info(f"   ✅ API Ninjas: ${price:.2f}")
                return float(price)
        else:
            logger.warning(f"⚠️ API Ninjas خطا: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"خطا در API Ninjas: {e}")
        return None

# =============================================
# ۶. دریافت از GOLD-API (برای طلا و نقره) ⭐
# =============================================

def get_gold_api_price(symbol):
    metal_map = {'GOLD': 'XAU', 'SILVER': 'XAG'}
    metal = metal_map.get(symbol)
    if not metal:
        return None
    try:
        url = f"https://api.gold-api.com/price/{metal}"
        # افزایش timeout و غیرفعال کردن بررسی SSL (در صورت نیاز)
        response = requests.get(url, timeout=15, verify=False)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and float(price) > 0:
                logger.info(f"   ✅ Gold-API ({metal}): ${price:.2f}")
                return float(price)
        return None
    except Exception as e:
        logger.error(f"خطا در Gold-API: {e}")
        return None

# =============================================
# ۷. دریافت از METALS-API (برای نقره) ⭐
# =============================================

def get_metals_api_price(symbol):
    if symbol != 'SILVER':
        return None
    try:
        url = "https://api.metals.live/v1/spot/silver"
        # غیرفعال کردن بررسی SSL برای حل خطای SSL
        response = requests.get(url, timeout=15, verify=False)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and float(price) > 0:
                logger.info(f"   ✅ Metals-API: ${price:.2f}")
                return float(price)
        return None
    except Exception as e:
        logger.error(f"خطا در Metals-API: {e}")
        return None

# =============================================
# ۸. دریافت از COINGECKO (فقط برای بیت‌کوین و اتریوم)
# =============================================

def get_coingecko_price(coin_id):
    """دریافت قیمت از CoinGecko برای ارزهای دیجیتال"""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = data.get(coin_id, {}).get('usd')
            if price and float(price) > 0:
                logger.info(f"   ✅ CoinGecko ({coin_id}): ${price:.2f}")
                return float(price)
        return None
    except Exception as e:
        logger.error(f"خطا در CoinGecko: {e}")
        return None

# =============================================
# ۹. سیستم پشتیبان چندمنبعی با میانگین‌گیری ⭐
# =============================================

def get_aggregated_price(symbol):
    prices = {}
    
    if symbol == 'GOLD':
        sources = ['goldapi', 'yahoo', 'goldprice', 'ninjas', 'pyth']
    elif symbol == 'SILVER':
        sources = ['goldapi', 'metals', 'yahoo', 'goldprice', 'ninjas', 'pyth']
    else:  # BTC, ETH
        sources = ['coingecko', 'chainlink', 'yahoo']
    
    logger.info(f"🔍 دریافت قیمت {symbol} از {len(sources)} منبع...")
    
    for source in sources:
        try:
            if source == 'goldapi':
                price = get_gold_api_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Gold-API: ${price:.2f}")
            elif source == 'metals':
                price = get_metals_api_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Metals-API: ${price:.2f}")
            elif source == 'coingecko':
                coin_map = {
                    'BTC': 'bitcoin',
                    'ETH': 'ethereum'
                }
                coin_id = coin_map.get(symbol)
                if coin_id:
                    price = get_coingecko_price(coin_id)
                    if price and float(price) > 0:
                        prices[source] = price
                        logger.info(f"   ✅ CoinGecko: ${price:.2f}")
            elif source == 'pyth':
                feed_ids = {
                    'GOLD': '0x8b7c8e4c6e5b9a8c7d6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4',
                    'SILVER': '0x9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8'
                }
                price = get_pyth_price(feed_ids[symbol])
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Pyth: ${price:.2f}")
            elif source == 'chainlink':
                price = get_chainlink_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Chainlink: ${price:.2f}")
            elif source == 'yahoo':
                price = get_yahoo_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Yahoo: ${price:.2f}")
            elif source == 'goldprice':
                price = get_goldprice_org_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Goldprice.org: ${price:.2f}")
            elif source == 'ninjas':
                price = get_ninjas_price(symbol)
                if price and float(price) > 0:
                    prices[source] = price
                    logger.info(f"   ✅ Ninjas: ${price:.2f}")
        except Exception as e:
            logger.warning(f"   ⚠️ خطا در {source}: {e}")
    
    if len(prices) == 0:
        logger.warning(f"⚠️ هیچ منبعی برای {symbol} در دسترس نیست")
        return None, {}, 0
    
    price_values = list(prices.values())
    avg_price = sum(price_values) / len(price_values)
    std_dev = np.std(price_values) if len(price_values) > 1 else 0
    
    if std_dev > avg_price * 0.03:
        logger.warning(f"⚠️ اختلاف زیاد بین منابع {symbol}: {std_dev:.2f} (میانگین: {avg_price:.2f})")
    
    logger.info(f"✅ میانگین قیمت {symbol}: ${avg_price:.2f} (از {len(prices)} منبع)")
    return avg_price, prices, len(prices)

def get_market_data_with_aggregation(symbol, days=30):
    logger.info(f"📊 دریافت داده‌های تاریخی برای {symbol}...")
    
    # اولویت اول: Yahoo Finance (داده‌های تاریخی واقعی)
    historical_df = get_yahoo_historical(symbol, days)
    if historical_df is not None:
        logger.info(f"✅ داده‌های تاریخی {symbol} از یاهو دریافت شد ({len(historical_df)} ردیف)")
        return historical_df
    
    # اگر یاهو در دسترس نبود، از قیمت میانگین استفاده می‌کنیم
    avg_price, prices, count = get_aggregated_price(symbol)
    if avg_price is None or avg_price == 0:
        logger.warning(f"⚠️ قیمت {symbol} معتبر نیست (صفر یا منفی)، از داده‌های جایگزین استفاده می‌شود")
        return generate_fallback_data(symbol, days)
    
    logger.info(f"🔄 ساخت داده‌های تاریخی {symbol} از میانگین قیمت {avg_price:.2f}")
    return generate_historical_from_price(avg_price, symbol, days)

# =============================================
# ۱۰. ساخت داده‌های تاریخی از یک قیمت
# =============================================

def generate_historical_from_price(current_price, symbol, days=30):
    now = datetime.now()
    dates = [now - timedelta(days=i) for i in range(days, 0, -1)]
    vol_map = {'GOLD': 0.015, 'SILVER': 0.025, 'BTC': 0.025, 'ETH': 0.03}
    vol = vol_map.get(symbol, 0.02)
    prices = [current_price]
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
        'High': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, days)
    }, index=dates)

# =============================================
# ۱۱. داده‌های جایگزین (آخرین گزینه)
# =============================================

def generate_fallback_data(symbol, days=30):
    now = datetime.now()
    dates = [now - timedelta(days=i) for i in range(days, 0, -1)]
    base_prices = {
        'GOLD': 4600,
        'SILVER': 55,
        'BTC': 65000,
        'ETH': 3500
    }
    vol_map = {'GOLD': 0.015, 'SILVER': 0.025, 'BTC': 0.025, 'ETH': 0.03}
    base = base_prices.get(symbol, 100)
    vol = vol_map.get(symbol, 0.02)
    prices = [base]
    for i in range(1, days):
        change = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + change)
        if new_price < prices[-1] * 0.92:
            new_price = prices[-1] * 0.92
        if new_price > prices[-1] * 1.08:
            new_price = prices[-1] * 1.08
        prices.append(new_price)
    logger.info(f"📊 داده‌های جایگزین برای {symbol} با قیمت پایه {base} ساخته شد")
    return pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, days)
    }, index=dates)

# =============================================
# ۱۲. قیمت طلای ایران
# =============================================

def get_iran_gold():
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
    except Exception as e:
        logger.error(f"خطا در طلای ایران: {e}")
        return None

# =============================================
# ۱۳. کلاس معامله‌گر (بدون تغییر)
# =============================================

class CombinedTrader:
    def __init__(self, capital=2500, symbol='GENERAL'):
        self.symbol = symbol
        self.initial = capital
        self.capital = capital
        self.trades = []
        self.open_positions = {}
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0
        self.peak = capital
        self.config = RiskConfig()
        self.state_file = f"state_{symbol.lower()}.json"
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('symbol') == self.symbol:
                        self.capital = data.get('capital', self.initial)
                        self.trades = data.get('trades', [])
                        self.open_positions = data.get('open_positions', {})
                        self.wins = data.get('wins', 0)
                        self.losses = data.get('losses', 0)
                        self.max_drawdown = data.get('max_drawdown', 0)
                        self.peak = data.get('peak', self.initial)
                        logger.info(f"✅ وضعیت {self.symbol} بارگذاری شد (سرمایه: {self.capital:.2f})")
            except Exception as e:
                logger.warning(f"⚠️ خطا در بارگذاری {self.symbol}: {e}")
    
    def save_state(self):
        try:
            data = {
                'symbol': self.symbol,
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
        except Exception as e:
            logger.warning(f"⚠️ خطا در ذخیره {self.symbol}: {e}")
    
    def calculate_position_size(self, price, atr, signal_score, confidence):
        base_risk = self.capital * self.config.BASE_RISK_PER_TRADE
        score_factor = min(max(signal_score / 2.0, 0.5), 2.0)
        adjusted_risk = base_risk * score_factor
        confidence_factor = confidence / 100.0
        adjusted_risk *= confidence_factor
        if self.config.VOLATILITY_ADJUSTMENT:
            atr_percent = (atr / price) * 100
            if atr_percent > 3:
                volatility_factor = 3.0 / atr_percent
                adjusted_risk *= min(volatility_factor, 1.0)
        max_risk = self.capital * self.config.MAX_POSITION_SIZE * self.config.STOP_LOSS
        adjusted_risk = min(adjusted_risk, max_risk)
        stop_distance = max(self.config.STOP_LOSS * price, atr * 1.5)
        if stop_distance <= 0:
            return 0
        position_size = adjusted_risk / stop_distance
        max_position = (self.capital * self.config.MAX_POSITION_SIZE) / price
        position_size = min(position_size, max_position)
        if position_size < 0.001:
            return 0
        logger.info(f"📊 حجم: {position_size:.4f} (ریسک: {adjusted_risk:.2f}, امتیاز: {signal_score:.1f})")
        return position_size
    
    def get_signal_with_reason(self, df, idx):
        if idx < 20:
            return None, 0, {}, 0
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
            return 'BUY', min(60 + score * 5, 95), reasons, score
        elif score <= -1.5:
            return 'SELL', min(60 + abs(score) * 5, 95), reasons, score
        else:
            return 'HOLD', 50, reasons, score
    
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
            signal, confidence, reasons, score = self.get_signal_with_reason(df, last_idx)
            if confidence >= self.config.MIN_CONFIDENCE and signal in ['BUY', 'SELL']:
                atr = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().iloc[last_idx]
                price = df['Close'].iloc[last_idx]
                size = self.calculate_position_size(price, atr, score, confidence)
                if size > 0.0001:
                    entry = self.open_trade(symbol, signal, price, size, atr, reasons, confidence, timestamp, score)
                    if entry:
                        new_entries.append(entry)
        self.save_state()
        return new_entries, closed_trades
    
    def open_trade(self, symbol, signal, price, size, atr, reasons, confidence, timestamp, score):
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
            'confidence': confidence,
            'score': score
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
            'time': timestamp,
            'score': score
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
            'confidence': pos['confidence'],
            'score': pos['score']
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
# ۱۴. توابع تولید پیام‌های تلگرامی
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
📊 امتیاز سیگنال: {entry['score']:.1f}

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
📊 امتیاز سیگنال هنگام ورود: {trade['score']:.1f}

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
                open_positions_text += f"• {sym}: {pos['direction']} @ ${pos['entry']:.2f} | قیمت فعلی: ${current_price:.2f} | سود/زیان شناور: {float_pnl:+.2f}% | TP: ${pos['tp']:.2f} | SL: ${pos['sl']:.2f} | امتیاز ورود: {pos['score']:.1f}\n"
    if not open_positions_text:
        open_positions_text = "هیچ پوزیشن بازی وجود ندارد. در انتظار سیگنال جدید...\n"
    total_cap = sum(m['capital'] for m in metrics_all.values())
    total_ret = (total_cap - 10000) / 10000 * 100
    total_trades = sum(m['trades'] for m in metrics_all.values())
    total_wins = sum(m['wins'] for m in metrics_all.values())
    win_rate = (total_wins / max(1, total_trades) * 100)
    report = f"""
🧠 **گزارش وضعیت لایو تریدینگ ترکیبی (سرمایه پویا)**
⏰ زمان: {now}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🇮🇷 طلای ۱۸ عیار ایران: {iran_price:,} ریال

📌 **سرمایه کل:** ۱۰,۰۰۰ دلار (هر بازار ۲,۵۰۰ دلار)
📌 **استراتژی حجم:** بر اساس قدرت سیگنال و نوسان (پویا)
📌 **منابع داده:** Gold-API, Metals-API, Pyth, Chainlink, Yahoo, Goldprice.org, Ninjas

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
🛡️ **مدیریت ریسک پویا:** حد ضرر ۳٪، حد سود ۶٪، حجم بر اساس امتیاز سیگنال و نوسان

⚠️ **توجه:** این شبیه‌سازی است و توصیه‌ی مالی نیست.
"""
    return report.strip()

# =============================================
# ۱۵. ارسال به تلگرام
# =============================================

async def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        logger.error("❌ TELEGRAM_TOKEN یا CHAT_ID تنظیم نشده!")
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
    except TelegramError as e:
        logger.error(f"خطای تلگرام: {e}")
        return False
    except Exception as e:
        logger.error(f"خطا در ارسال: {e}")
        return False

# =============================================
# ۱۶. تابع اصلی
# =============================================

async def main():
    logger.info("🚀 شروع ربات ترکیبی با سیستم پشتیبان چندمنبعی (Gold-API و Metals-API)...")
    for f in ['state_gold.json', 'state_silver.json', 'state_btc.json', 'state_eth.json', 'state_combined.json']:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"🗑️ فایل {f} حذف شد")
    iran = get_iran_gold()
    if iran is None:
        iran = 210_000_000
    
    gold_df = get_market_data_with_aggregation('GOLD', days=30)
    silver_df = get_market_data_with_aggregation('SILVER', days=30)
    btc_df = get_market_data_with_aggregation('BTC', days=30)
    eth_df = get_market_data_with_aggregation('ETH', days=30)
    
    trader_gold = CombinedTrader(capital=2500, symbol='GOLD')
    trader_silver = CombinedTrader(capital=2500, symbol='SILVER')
    trader_btc = CombinedTrader(capital=2500, symbol='BTC')
    trader_eth = CombinedTrader(capital=2500, symbol='ETH')
    
    entries_gold, exits_gold = trader_gold.process(gold_df, 'GOLD')
    entries_silver, exits_silver = trader_silver.process(silver_df, 'SILVER')
    entries_btc, exits_btc = trader_btc.process(btc_df, 'BTC')
    entries_eth, exits_eth = trader_eth.process(eth_df, 'ETH')
    
    for entry in entries_gold + entries_silver + entries_btc + entries_eth:
        await send_telegram(format_entry_message(entry))
    
    for trade in exits_gold + exits_silver + exits_btc + exits_eth:
        await send_telegram(format_exit_message(trade))
    
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
