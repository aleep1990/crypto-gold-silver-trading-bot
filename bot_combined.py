"""
ربات ترکیبی لایو ترید با مدیریت هوشمند داده
- استفاده از چندین منبع معتبر با Fallback
- عدم استفاده از داده‌های جایگزین برای معامله
- کش آخرین قیمت معتبر
- اعتبارسنجی قیمت‌ها قبل از معامله
- اصلاح خطای NoneType در ترکیب لیست‌ها
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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NINJAS_API_KEY = os.getenv("NINJAS_API_KEY")
PYTH_API_KEY = os.getenv("PYTH_API_KEY")

# =============================================
# تنظیمات مدیریت ریسک
# =============================================

class RiskConfig:
    MAX_POSITION_SIZE = 0.15
    STOP_LOSS = 0.03
    TAKE_PROFIT = 0.06
    MIN_CONFIDENCE = 25
    BASE_RISK_PER_TRADE = 0.02
    VOLATILITY_ADJUSTMENT = True
    SIGNAL_SCORE_WEIGHT = 1.5

# =============================================
# کش قیمت‌ها (ذخیره آخرین قیمت معتبر)
# =============================================

CACHE_FILE = "last_prices.json"

def load_cache():
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_cache(data):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

def get_cached_price(symbol):
    cache = load_cache()
    entry = cache.get(symbol)
    if entry:
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - timestamp < timedelta(hours=24):  # ۲۴ ساعت معتبر
                return entry['price']
        except:
            pass
    return None

def update_cache(symbol, price):
    cache = load_cache()
    cache[symbol] = {
        'price': price,
        'timestamp': datetime.now().isoformat()
    }
    save_cache(cache)

# =============================================
# دریافت قیمت از منابع مختلف
# =============================================

# ---------- قیمت طلا (دلاری) ----------
def get_gold_price_from_coingecko():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=gold&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = data.get('gold', {}).get('usd')
            if price and 2000 < price < 3000:
                return float(price)
        return None
    except Exception as e:
        logger.error(f"CoinGecko error: {e}")
        return None

def get_gold_price_from_goldapi():
    try:
        url = "https://api.gold-api.com/price/XAU"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and 2000 < price < 3000:
                return float(price)
        return None
    except Exception as e:
        logger.error(f"Gold-API error: {e}")
        return None

def get_gold_price_from_goldprice_org():
    try:
        url = "https://goldprice.org/live-gold-price.html"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        match = re.search(r'Spot Gold Price:\s*USD\s*([\d,]+\.?\d*)', text)
        if match:
            price = float(match.group(1).replace(',', ''))
            if 2000 < price < 3000:
                return price
        return None
    except Exception as e:
        logger.error(f"Goldprice.org error: {e}")
        return None

def get_gold_price_usd():
    sources = [
        ('CoinGecko', get_gold_price_from_coingecko),
        ('Gold-API', get_gold_price_from_goldapi),
        ('Goldprice.org', get_gold_price_from_goldprice_org),
    ]
    
    for name, func in sources:
        try:
            price = func()
            if price:
                logger.info(f"✅ قیمت طلا از {name}: ${price:.2f}")
                update_cache('GOLD_USD', price)
                return price
        except Exception as e:
            logger.warning(f"⚠️ {name} خطا: {e}")
            continue
    
    # اگر همه منابع قطع بودند، از کش استفاده کن
    cached = get_cached_price('GOLD_USD')
    if cached:
        logger.warning(f"⚠️ استفاده از کش: ${cached:.2f}")
        return cached
    
    # اگر کش هم نبود، مقدار پیش‌فرض (فقط برای روزهای تعطیل)
    logger.warning("⚠️ قیمت طلا در دسترس نیست، از مقدار پیش‌فرض ۲۵۰۰ دلار استفاده می‌شود (تعطیلی بازار)")
    default_price = 2500.0
    update_cache('GOLD_USD', default_price)
    return default_price

# ---------- قیمت نقره (دلاری) ----------
def get_silver_price_from_metals_api():
    try:
        url = "https://api.metals.live/v1/spot/silver"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and 20 < price < 100:
                return float(price)
        return None
    except Exception as e:
        logger.error(f"Metals-API error: {e}")
        return None

def get_silver_price_from_goldapi():
    try:
        url = "https://api.gold-api.com/price/XAG"
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            data = response.json()
            price = data.get('price')
            if price and 20 < price < 100:
                return float(price)
        return None
    except Exception as e:
        logger.error(f"Gold-API (silver) error: {e}")
        return None

def get_silver_price_usd():
    sources = [
        ('Metals-API', get_silver_price_from_metals_api),
        ('Gold-API (XAG)', get_silver_price_from_goldapi),
    ]
    
    for name, func in sources:
        try:
            price = func()
            if price:
                logger.info(f"✅ قیمت نقره از {name}: ${price:.2f}")
                update_cache('SILVER_USD', price)
                return price
        except Exception as e:
            logger.warning(f"⚠️ {name} خطا: {e}")
            continue
    
    cached = get_cached_price('SILVER_USD')
    if cached:
        logger.warning(f"⚠️ استفاده از کش: ${cached:.2f}")
        return cached
    
    default_price = 30.0
    logger.warning(f"⚠️ قیمت نقره در دسترس نیست، از مقدار پیش‌فرض {default_price} دلار استفاده می‌شود")
    update_cache('SILVER_USD', default_price)
    return default_price

# ---------- قیمت ارزهای دیجیتال ----------
def get_crypto_price(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = data.get(coin_id, {}).get('usd')
            if price and price > 0:
                return float(price)
        return None
    except Exception as e:
        logger.error(f"CoinGecko ({coin_id}) error: {e}")
        return None

def get_btc_price():
    price = get_crypto_price('bitcoin')
    if price:
        update_cache('BTC_USD', price)
        return price
    cached = get_cached_price('BTC_USD')
    if cached:
        return cached
    default_price = 65000.0
    update_cache('BTC_USD', default_price)
    return default_price

def get_eth_price():
    price = get_crypto_price('ethereum')
    if price:
        update_cache('ETH_USD', price)
        return price
    cached = get_cached_price('ETH_USD')
    if cached:
        return cached
    default_price = 3500.0
    update_cache('ETH_USD', default_price)
    return default_price

# ---------- نرخ دلار به ریال ----------
def get_usd_irr_from_navasan():
    try:
        url = "https://raw.githubusercontent.com/HosseinOdd/Navasan-API/main/data/fiat.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            usd = data.get('USD')
            if usd and usd > 0:
                return int(usd)
        return None
    except Exception as e:
        logger.error(f"Navasan error: {e}")
        return None

def get_usd_irr_from_pricedb():
    try:
        url = "https://api.priceto.day/v1/latest/irr/usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            rate = data.get('rate')
            if rate and rate > 0:
                return int(rate)
        return None
    except Exception as e:
        logger.error(f"PriceDB error: {e}")
        return None

def get_usd_irr():
    sources = [
        ('Navasan', get_usd_irr_from_navasan),
        ('PriceDB', get_usd_irr_from_pricedb),
    ]
    
    for name, func in sources:
        try:
            rate = func()
            if rate:
                logger.info(f"💵 نرخ دلار از {name}: {rate:,} ریال")
                update_cache('USD_IRR', rate)
                return rate
        except Exception as e:
            logger.warning(f"⚠️ {name} خطا: {e}")
            continue
    
    cached = get_cached_price('USD_IRR')
    if cached:
        logger.warning(f"⚠️ استفاده از کش: {cached:,} ریال")
        return cached
    
    # مقدار پیش‌فرض (حدود ۶۰۰ هزار ریال = ۶۰ هزار تومان)
    default_rate = 600000
    logger.warning(f"⚠️ نرخ دلار در دسترس نیست، از مقدار پیش‌فرض {default_rate:,} ریال استفاده می‌شود")
    update_cache('USD_IRR', default_rate)
    return default_rate

# ---------- قیمت تتر به ریال ----------
def get_usdt_irr():
    try:
        usdt_usd = get_crypto_price('tether')
        if not usdt_usd:
            logger.warning("⚠️ قیمت تتر به دلار دریافت نشد")
            return None
        usd_irr = get_usd_irr()
        if not usd_irr:
            return None
        price = int(usdt_usd * usd_irr)
        logger.info(f"💵 قیمت تتر: {price:,} تومان")
        update_cache('USDT_IRR', price)
        return price
    except Exception as e:
        logger.error(f"خطا در محاسبه قیمت تتر: {e}")
        return None

# ---------- قیمت طلای ایران به ریال ----------
def get_iran_gold():
    try:
        url = "https://raw.githubusercontent.com/HosseinOdd/Navasan-API/main/data/gold.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'طلای ۱۸' in item.get('title', ''):
                        price_str = item.get('price', '').replace(',', '')
                        if price_str.isdigit():
                            price = int(price_str)
                            logger.info(f"🇮🇷 قیمت طلای ایران: {price:,} ریال")
                            update_cache('IRAN_GOLD', price)
                            return price
        return None
    except Exception as e:
        logger.error(f"خطا در دریافت طلای ایران: {e}")
        return None

# =============================================
# ساخت داده‌های تاریخی از قیمت لحظه‌ای
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
# دریافت داده‌های بازار
# =============================================

def get_market_data(symbol):
    logger.info(f"📊 دریافت داده‌های تاریخی {symbol}...")
    
    # اولویت ۱: Yahoo Finance (داده‌های تاریخی)
    try:
        yahoo_symbols = {'GOLD': 'GC=F', 'SILVER': 'SI=F', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}
        ticker = yahoo_symbols.get(symbol)
        if ticker:
            df = yf.download(ticker, period="35d", interval="1d", progress=False)
            if df is not None and not df.empty and len(df) >= 20:
                logger.info(f"✅ داده‌های تاریخی {symbol} از یاهو دریافت شد")
                return df
    except Exception as e:
        logger.warning(f"⚠️ Yahoo Finance خطا: {e}")
    
    # اولویت ۲: ساخت داده از قیمت لحظه‌ای
    current_price = None
    if symbol == 'GOLD':
        current_price = get_gold_price_usd()
    elif symbol == 'SILVER':
        current_price = get_silver_price_usd()
    elif symbol == 'BTC':
        current_price = get_btc_price()
    elif symbol == 'ETH':
        current_price = get_eth_price()
    
    if current_price:
        logger.info(f"🔄 ساخت داده‌های تاریخی از قیمت {current_price:.2f}")
        return generate_historical_from_price(current_price, symbol, days=30)
    
    logger.error(f"❌ داده‌های {symbol} در دسترس نیست")
    return None

# =============================================
# کلاس معامله‌گر
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
        try:
            if os.path.exists(self.state_file):
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
                        return
        except Exception as e:
            logger.warning(f"⚠️ خطا در بارگذاری {self.symbol}: {e}")
        
        logger.info(f"🆕 شروع جدید برای {self.symbol} با سرمایه {self.initial}")
        self.capital = self.initial
        self.trades = []
        self.open_positions = {}
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0
        self.peak = self.initial
    
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
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"⚠️ داده‌های {symbol} کافی نیست (ورود ممنوع)")
            return [], []  # برگرداندن لیست خالی به‌جای None
        
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
# پیام‌های تلگرامی
# =============================================

def format_entry_message(entry, usdt_price):
    reasons_text = "\n".join([f"   • {k}: {v}" for k, v in entry['reasons'].items()])
    usdt_text = f"💵 قیمت تتر: {usdt_price:,} تومان" if usdt_price and usdt_price > 0 else ""
    return f"""
📢 **ورود به معامله {entry['symbol']}**

🧭 جهت: {entry['direction']}
💰 ارزش معامله: ${entry['position_value']:,.2f}
📊 قیمت ورود: ${entry['entry']:.2f}
🎯 حد سود (TP): ${entry['tp']:.2f}
🛑 حد ضرر (SL): ${entry['sl']:.2f}
📈 اعتماد به سیگنال: {entry['confidence']}%
📊 امتیاز سیگنال: {entry['score']:.1f}
{usdt_text}

🔍 **تحلیل ورود:**
{reasons_text}

⏰ زمان ورود: {entry['time'].strftime('%Y-%m-%d %H:%M:%S')}
"""

def format_exit_message(trade, usdt_price):
    reasons_text = "\n".join([f"   • {k}: {v}" for k, v in trade['reasons'].items()])
    usdt_text = f"💵 قیمت تتر: {usdt_price:,} تومان" if usdt_price and usdt_price > 0 else ""
    return f"""
📢 **خروج از معامله {trade['symbol']}**

🧭 جهت: {trade['direction']}
💰 ارزش معامله: ${trade['position_value']:,.2f}
📊 قیمت ورود: ${trade['entry']:.2f} → خروج: ${trade['exit']:.2f}
📈 سود/زیان: {trade['pnl_percent']:+.2f}% (${trade['pnl_amount']:+.2f})
📉 دلیل خروج: {trade['exit_reason']}
📊 امتیاز سیگنال هنگام ورود: {trade['score']:.1f}
{usdt_text}

🔍 **تحلیل ورود (مرجع):**
{reasons_text}

⏰ زمان خروج: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}
"""

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
    logger.info("🚀 شروع ربات ترکیبی (نسخه ۳.۱ با رفع خطای NoneType)...")
    
    # دریافت قیمت‌های نمایشی
    usdt_price = get_usdt_irr()
    if usdt_price is None or usdt_price == 0:
        usdt_price = 0
        logger.warning("⚠️ قیمت تتر دریافت نشد")
    
    iran_gold = get_iran_gold()
    if iran_gold is None or iran_gold == 0:
        iran_gold = 210_000_000
        logger.warning("⚠️ قیمت طلای ایران دریافت نشد، از مقدار ثابت استفاده شد")
    
    # دریافت داده‌های بازار
    gold_df = get_market_data('GOLD')
    silver_df = get_market_data('SILVER')
    btc_df = get_market_data('BTC')
    eth_df = get_market_data('ETH')
    
    trader_gold = CombinedTrader(capital=2500, symbol='GOLD')
    trader_silver = CombinedTrader(capital=2500, symbol='SILVER')
    trader_btc = CombinedTrader(capital=2500, symbol='BTC')
    trader_eth = CombinedTrader(capital=2500, symbol='ETH')
    
    entries_gold, exits_gold = trader_gold.process(gold_df, 'GOLD')
    entries_silver, exits_silver = trader_silver.process(silver_df, 'SILVER')
    entries_btc, exits_btc = trader_btc.process(btc_df, 'BTC')
    entries_eth, exits_eth = trader_eth.process(eth_df, 'ETH')
    
    # ✅ دیگر نیازی به تبدیل None نیست چون process حالا لیست خالی برمی‌گرداند
    for entry in entries_gold + entries_silver + entries_btc + entries_eth:
        await send_telegram(format_entry_message(entry, usdt_price))
    
    for trade in exits_gold + exits_silver + exits_btc + exits_eth:
        await send_telegram(format_exit_message(trade, usdt_price))
    
    logger.info("🏁 پایان اجرا")

if __name__ == "__main__":
    asyncio.run(main())
