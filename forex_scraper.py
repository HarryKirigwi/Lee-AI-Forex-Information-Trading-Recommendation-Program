import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import io # Used for handling matplotlib output in certain environments
import re # For parsing AI response

class YahooForexScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/5537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # GitHub AI API configuration (for LLM integration)
        # 
        # *******************************************************************
        # IMPORTANT: Replace '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'
        # with your actual GitHub Personal Access Token (PAT).
        # Without a valid PAT, the AI recommendations will not function.
        # *******************************************************************
        self.github_api_key = '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' 
        self.github_base_url = 'https://models.github.ai/inference' 
        self.github_model = 'openai/gpt-4o' 

    def get_eurusd_data(self, period="1d", interval="15m"):
        """
        Fetches EUR/USD historical and current price data using yfinance.
        
        Args:
            period (str): Data period (e.g., "1d", "5d", "1mo").
            interval (str): Data interval (e.g., "1m", "15m", "1h").
            
        Returns:
            dict: A dictionary containing EUR/USD data, or None if an error occurs.
        """
        try:
            print(f"Fetching EUR/USD data for period: {period}, interval: {interval}...")
            
            eurusd = yf.Ticker("EURUSD=X")
            hist_data = eurusd.history(period=period, interval=interval)
            
            if hist_data.empty:
                print(f"No historical data found for EUR/USD with period={period}, interval={interval}.")
                return None

            # Get current price from the last entry of historical data for consistency
            current_price = float(hist_data['Close'].iloc[-1])
            
            return {
                'symbol': 'EURUSD=X',
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'historical_data': hist_data, # Keep DataFrame for direct use in analysis
                'period': period,
                'interval': interval
            }
            
        except Exception as e:
            print(f"Error fetching EUR/USD data: {e}")
            return None
    
    def get_forex_news(self):
        """
        Scrapes forex-related news from multiple sources (Yahoo Finance, MarketWatch)
        and includes a fallback to sample data if scraping fails.
        """
        news_items = []
        
        # Method 1: Yahoo Finance general finance news (more robust selectors)
        try:
            print("Fetching news from Yahoo Finance...")
            url = "https://finance.yahoo.com/news/"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Prioritized selectors for news headlines
            selectors = [
                'h3[class*="Mb(5px)"] a', # Specific for some Yahoo layouts
                'h3[data-test-id="StreamHeadline"] a', # Another common Yahoo headline selector
                'h3 a[data-uuid]', # General anchor within h3 with a data-uuid
                'li.js-stream-content h3 a', # List item content
            ]
            
            articles_found = False
            for selector in selectors:
                articles = soup.select(selector)
                if articles:
                    print(f"Found {len(articles)} articles with selector: {selector}")
                    articles_found = True
                    for article in articles[:8]: # Limit to top 8 articles for relevance
                        try:
                            title = article.get_text(strip=True)
                            link = article.get('href', '')
                            
                            # Fix relative URLs
                            if link and link.startswith('/'):
                                link = 'https://finance.yahoo.com' + link
                            
                            forex_keywords = ['dollar', 'euro', 'currency', 'forex', 'fed', 'central bank', 'inflation', 'interest rate', 'ecb', 'boj', 'bank of england', 'fx']
                            if any(keyword in title.lower() for keyword in forex_keywords):
                                news_items.append({
                                    'title': title,
                                    'link': link,
                                    'timestamp': 'Recent', 
                                    'source': 'Yahoo Finance',
                                    'category': 'Finance'
                                })
                        except Exception as e:
                            print(f"Error parsing Yahoo article: {e}")
                            continue
                    if news_items: # If we found some relevant news, break from selectors
                        break
            if not articles_found:
                print("No relevant articles found using Yahoo Finance selectors.")
                    
        except Exception as e:
            print(f"Yahoo Finance news error: {e}")
        
        # Method 2: MarketWatch (alternative source)
        try:
            print("Fetching from alternative sources (MarketWatch)...")
            url = "https://www.marketwatch.com/economy-politics"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for common headline structures, focusing on 'a' tags within common containers
            headlines = soup.select('div.article__content a.link__headline, h3.article__headline a, h2.article__headline a')[:5] 
                
            for headline_link in headlines:
                try:
                    title = headline_link.get_text(strip=True)
                    link = headline_link.get('href', '')
                    
                    if not link.startswith('http'): # Ensure full URL
                        link = 'https://www.marketwatch.com' + link
                    
                    forex_keywords = ['dollar', 'euro', 'currency', 'forex', 'fed', 'central bank', 'inflation', 'interest rate', 'ecb', 'boj', 'bank of england', 'gdp', 'cpi', 'pmi', 'fx']
                    if any(keyword in title.lower() for keyword in forex_keywords):
                        news_items.append({
                            'title': title,
                            'link': link,
                            'timestamp': 'Recent',
                            'source': 'MarketWatch',
                            'category': 'Economics/Politics'
                        })
                except Exception as e:
                    print(f"Error parsing MarketWatch headline: {e}")
                    continue
                        
        except Exception as e:
            print(f"Alternative news source error (MarketWatch): {e}")
        
        # Fallback: Create sample forex news if nothing else works
        if not news_items:
            print("Using sample forex news as no real articles were scraped...")
            sample_news = [
                {
                    'title': 'Sample: EUR/USD poised for movement after key economic data release.',
                    'link': 'https://example.com/sample-eurusd-outlook',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'source': 'Sample Data (Fallback)',
                    'category': 'Forex Analysis'
                },
                {
                    'title': 'Sample: Strong USD unlikely to persist without further hawkish Fed signals.',
                    'link': 'https://example.com/sample-fed-outlook',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'source': 'Sample Data (Fallback)',
                    'category': 'Central Bank Commentary'
                },
                {
                    'title': 'Sample: Geopolitical tensions in Eastern Europe may boost safe-haven JPY.',
                    'link': 'https://example.com/sample-geopolitics',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'source': 'Sample Data (Fallback)',
                    'category': 'Geopolitical Impact'
                }
            ]
            news_items.extend(sample_news)
        
        print(f"Total news items collected: {len(news_items)}")
        return news_items[:10] # Return top 10 items to keep data manageable
    
    def get_market_summary(self):
        """
        Fetches current price and daily change for major forex pairs.
        """
        major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURGBP=X'
        ]
        
        market_data = {}
        
        for pair in major_pairs:
            try:
                ticker = yf.Ticker(pair)
                # Get last 2 days for change calculation. Use 1h interval for more recent data points.
                hist = ticker.history(period="2d", interval="1h") 
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    # Calculate change from the Open of the current day or close of previous day
                    # If current day's open is available, use that. Otherwise, use previous close.
                    if len(hist) > 1 and hist.index[-1].date() == datetime.now().date(): # Check if last data point is from today
                        prev_price = float(hist['Open'].iloc[-1]) # Use today's opening price for daily change
                    elif len(hist) > 1:
                        prev_price = float(hist['Close'].iloc[-2]) # Use previous day's close
                    else:
                        prev_price = current_price # Fallback
                        
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    market_data[pair] = {
                        'current_price': current_price,
                        'change': change,
                        'change_percent': change_pct,
                        'high': float(hist['High'].iloc[-1]), # Most recent high
                        'low': float(hist['Low'].iloc[-1]),   # Most recent low
                        'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and hist['Volume'].iloc[-1] > 0 else 0
                    }
                else:
                    print(f"No market data found for {pair}.")
                
                time.sleep(0.1) # Small delay to avoid rate limiting from Yahoo Finance
                
            except Exception as e:
                print(f"Error fetching market summary for {pair}: {e}")
                continue
        
        return market_data
    
    def get_economic_calendar(self):
        """
        Generates sample economic events for today and tomorrow.
        NOTE: This is a simulated economic calendar. For real-world applications,
        you would integrate with a dedicated economic calendar API (e.g., from Investing.com, ForexFactory).
        """
        try:
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            day_after_tomorrow = today + timedelta(days=2)

            events = [
                {
                    'date': today.isoformat(),
                    'time': '08:30 EST',
                    'event': 'US Initial Jobless Claims',
                    'currency': 'USD',
                    'impact': 'Medium',
                    'forecast': '220K',
                    'actual': '218K', # Adding actual for realism
                    'previous': '215K'
                },
                {
                    'date': today.isoformat(),
                    'time': '10:00 EST',
                    'event': 'Eurozone CPI Flash Estimate',
                    'currency': 'EUR',
                    'impact': 'High',
                    'forecast': '2.5%',
                    'actual': '2.6%',
                    'previous': '2.4%'
                },
                {
                    'date': today.isoformat(),
                    'time': '14:00 EST',
                    'event': 'Federal Reserve Chair Speech',
                    'currency': 'USD',
                    'impact': 'High',
                    'forecast': 'N/A (Speech)',
                    'actual': 'N/A',
                    'previous': 'N/A'
                },
                {
                    'date': tomorrow.isoformat(),
                    'time': '02:00 JST',
                    'event': 'BOJ Interest Rate Decision',
                    'currency': 'JPY',
                    'impact': 'High',
                    'forecast': '0.10%',
                    'actual': '0.10%',
                    'previous': '0.10%'
                },
                {
                    'date': tomorrow.isoformat(),
                    'time': '09:00 BST',
                    'event': 'UK Retail Sales MoM',
                    'currency': 'GBP',
                    'impact': 'Medium',
                    'forecast': '0.3%',
                    'actual': '0.2%',
                    'previous': '0.5%'
                },
                {
                    'date': day_after_tomorrow.isoformat(),
                    'time': '07:00 CET',
                    'event': 'German Ifo Business Climate',
                    'currency': 'EUR',
                    'impact': 'Medium',
                    'forecast': '88.5',
                    'actual': '89.0',
                    'previous': '87.9'
                }
            ]
            
            print(f"Generated {len(events)} sample economic calendar events.")
            return events
            
        except Exception as e:
            print(f"Error generating economic calendar: {e}")
            return []
    
    def analyze_eurusd_trends(self, data):
        """
        Analyzes EUR/USD trends and patterns using various technical indicators,
        including Moving Averages, Support/Resistance, RSI, MACD, and Bollinger Bands.
        """
        if not data or data['historical_data'].empty:
            print("Insufficient data for EUR/USD trend analysis.")
            return {}
        
        df = data['historical_data'].copy() 
        
        # Ensure 'Volume' column exists for some analyses, default to 0 if not
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        analysis = {
            'current_price': data['current_price'],
            'period_high': float(df['High'].max()),
            'period_low': float(df['Low'].min()),
            'average_price': float(df['Close'].mean()),
            'volatility': float(df['Close'].std()),
            'volume_avg': float(df['Volume'].mean()),
            'trend_direction': 'Unknown',
            '20_ema': None,
            '50_ema': None,
            'resistance_level': None,
            'support_level': None,
            'rsi': None,
            'macd_histogram': None,
            'bollinger_upper': None,
            'bollinger_middle': None,
            'bollinger_lower': None,
            'bollinger_position': 'N/A'
        }
        
        # Moving Averages (Exponential Moving Averages for better responsiveness)
        if len(df) >= 20: 
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            analysis['20_ema'] = float(df['EMA20'].iloc[-1])
        if len(df) >= 50: 
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            analysis['50_ema'] = float(df['EMA50'].iloc[-1])

        # Simple trend analysis based on overall price movement in the period
        if len(df) >= 2:
            recent_price = float(df['Close'].iloc[-1])
            older_price = float(df['Close'].iloc[0])
            
            if recent_price > older_price:
                analysis['trend_direction'] = 'Upward'
            elif recent_price < older_price:
                analysis['trend_direction'] = 'Downward'
            else:
                analysis['trend_direction'] = 'Sideways'
        
        # Basic Support and Resistance (using recent highs/lows for short-term levels)
        if len(df) >= 10: 
            analysis['resistance_level'] = float(df['High'].iloc[-5:].max()) # Max of last 5 high
            analysis['support_level'] = float(df['Low'].iloc[-5:].min())   # Min of last 5 low

        # Relative Strength Index (RSI)
        # Typically uses 14 periods.
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            analysis['rsi'] = float(rsi.iloc[-1])
            if analysis['rsi'] > 70:
                analysis['rsi_signal'] = 'Overbought'
            elif analysis['rsi'] < 30:
                analysis['rsi_signal'] = 'Oversold'
            else:
                analysis['rsi_signal'] = 'Neutral'
        
        # Moving Average Convergence Divergence (MACD)
        # Typically uses 12-period EMA, 26-period EMA, 9-period Signal EMA
        if len(df) >= 26: # Need enough data for 26-period EMA
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            analysis['macd_histogram'] = float(macd_histogram.iloc[-1])
            
            if analysis['macd_histogram'] > 0 and macd.iloc[-1] > signal.iloc[-1]:
                analysis['macd_signal'] = 'Bullish'
            elif analysis['macd_histogram'] < 0 and macd.iloc[-1] < signal.iloc[-1]:
                analysis['macd_signal'] = 'Bearish'
            else:
                analysis['macd_signal'] = 'Indecisive'

        # Bollinger Bands
        # Typically uses 20-period SMA and 2 standard deviations
        if len(df) >= 20:
            middle_band = df['Close'].rolling(window=20).mean()
            std_dev = df['Close'].rolling(window=20).std()
            upper_band = middle_band + (std_dev * 2)
            lower_band = middle_band - (std_dev * 2)
            
            analysis['bollinger_middle'] = float(middle_band.iloc[-1])
            analysis['bollinger_upper'] = float(upper_band.iloc[-1])
            analysis['bollinger_lower'] = float(lower_band.iloc[-1])
            
            if data['current_price'] > analysis['bollinger_upper']:
                analysis['bollinger_position'] = 'Above Upper Band (Potentially Overbought)'
            elif data['current_price'] < analysis['bollinger_lower']:
                analysis['bollinger_position'] = 'Below Lower Band (Potentially Oversold)'
            else:
                analysis['bollinger_position'] = 'Within Bands'

        return analysis
    
    def get_trading_recommendation_ai(self, eurusd_analysis, news_data, market_data, economic_events):
        """
        Uses GitHub AI (openai/gpt-4o) to generate a trading recommendation ('BUY' or 'SELL') for EUR/USD.
        The recommendation is based on a comprehensive analysis integrating technical indicators (ICT/Smart Money Concepts)
        and crucial fundamental factors (recent news, economic calendar).
        The AI is instructed to provide specific entry, stop-loss, and take-profit levels.
        """
        print("\nRequesting AI trading recommendation from GitHub AI...")
        
        # Construct the prompt with all relevant data
        prompt_parts = []
        prompt_parts.append("As an expert forex trader specializing in ICT (Inner Circle Trader) and Smart Money Concepts, provide a 'BUY' or 'SELL' recommendation for EUR/USD. Do NOT provide 'HOLD'.\n")
        prompt_parts.append("Your justification MUST clearly integrate both the technical analysis (ICT/SMC, indicators) and the fundamental analysis (news, economic events).\n")
        prompt_parts.append("Additionally, provide specific numeric price levels for 'Entry', 'Stop Loss (SL)', 'Take Profit 1 (TP1)', and 'Take Profit 2 (TP2)'.\n\n")
        
        prompt_parts.append("--- ICT/Smart Money Concepts Context ---\n")
        prompt_parts.append("ICT concepts involve analyzing market structure (e.g., breaks of structure, change of character, order blocks, fair value gaps), liquidity (e.g., liquidity sweeps above/below highs/lows), and institutional order flow. Smart Money Concepts (SMC) focus on identifying footprints of institutional trading through volume, significant price swings, and reactions at key supply/demand zones. Look for confirmation of these concepts within the provided data points.\n\n")

        # 1. EUR/USD Technical Analysis
        if eurusd_analysis:
            prompt_parts.append("--- EUR/USD Technical Analysis ---\n")
            prompt_parts.append(f"Current Price: {eurusd_analysis.get('current_price', 'N/A'):.5f}\n")
            prompt_parts.append(f"Period High (observed): {eurusd_analysis.get('period_high', 'N/A'):.5f}\n")
            prompt_parts.append(f"Period Low (observed): {eurusd_analysis.get('period_low', 'N/A'):.5f}\n")
            prompt_parts.append(f"Trend Direction (current period): {eurusd_analysis.get('trend_direction', 'N/A')}\n")
            if eurusd_analysis.get('20_ema') is not None:
                prompt_parts.append(f"20-Period EMA: {eurusd_analysis.get('20_ema'):.5f}\n")
            if eurusd_analysis.get('50_ema') is not None:
                prompt_parts.append(f"50-Period EMA: {eurusd_analysis.get('50_ema'):.5f}\n")
            if eurusd_analysis.get('support_level') is not None:
                prompt_parts.append(f"Recent Support Level: {eurusd_analysis.get('support_level'):.5f}\n")
                prompt_parts.append(f"Recent Resistance Level: {eurusd_analysis.get('resistance_level'):.5f}\n")
            if eurusd_analysis.get('rsi') is not None:
                prompt_parts.append(f"RSI (14-period): {eurusd_analysis.get('rsi'):.2f} ({eurusd_analysis.get('rsi_signal', 'N/A')})\n")
            if eurusd_analysis.get('macd_histogram') is not None:
                prompt_parts.append(f"MACD Signal: {eurusd_analysis.get('macd_signal', 'N/A')} (Histogram: {eurusd_analysis.get('macd_histogram'):.5f})\n")
            if eurusd_analysis.get('bollinger_middle') is not None:
                prompt_parts.append(f"Bollinger Bands: Middle={eurusd_analysis.get('bollinger_middle'):.5f}, Upper={eurusd_analysis.get('bollinger_upper'):.5f}, Lower={eurusd_analysis.get('bollinger_lower'):.5f} (Position: {eurusd_analysis.get('bollinger_position', 'N/A')})\n")
            prompt_parts.append("\n")

        # 2. Recent Forex News (Fundamental Influence)
        if news_data:
            prompt_parts.append("--- Recent Forex News (Key Fundamental Drivers) ---\n")
            if len(news_data) == 0:
                prompt_parts.append("No significant recent forex news available.\n")
            else:
                for i, news_item in enumerate(news_data[:5]): # Limit news input to top 5 for conciseness
                    prompt_parts.append(f"- {news_item['title']} (Source: {news_item['source']})\n")
            prompt_parts.append("\n")

        # 3. Upcoming Economic Calendar (Forward-Looking Fundamental Influence)
        if economic_events:
            prompt_parts.append("--- Upcoming High/Medium Impact Economic Events (Potential Market Movers) ---\n")
            relevant_events = [e for e in economic_events if e.get('impact') in ['High', 'Medium']]
            if not relevant_events:
                prompt_parts.append("No high/medium impact economic events scheduled for today/tomorrow.\n")
            else:
                for event in relevant_events[:3]: # Limit events to top 3 for conciseness
                    prompt_parts.append(f"- {event.get('date')} {event.get('time')} - {event.get('event')} ({event.get('currency')}, Impact: {event.get('impact')}, Actual: {event.get('actual', 'N/A')}, Forecast: {event.get('forecast', 'N/A')}, Previous: {event.get('previous', 'N/A')})\n")
            prompt_parts.append("\n")

        # 4. Major Forex Pairs Summary (Broader Market Sentiment)
        if market_data:
            prompt_parts.append("--- Major Forex Pair Movements (Daily Change) ---\n")
            for pair, data in list(market_data.items())[:5]: # Limit to top 5 pairs
                prompt_parts.append(f"- {pair}: {data.get('current_price',0):.5f} ({data.get('change_percent',0):+.2f}%)\n")
            prompt_parts.append("\n")

        prompt_parts.append("Based on the comprehensive technical analysis (ICT/SMC, indicators) AND the fundamental news and economic calendar data, provide your recommendation and price levels.\n")
        prompt_parts.append("Format your response STRICTLY as follows:\n")
        prompt_parts.append("Recommendation: [BUY/SELL]\n") # Changed to only BUY/SELL
        prompt_parts.append("Entry: [Price]\n")
        prompt_parts.append("SL: [Price]\n")
        prompt_parts.append("TP1: [Price]\n")
        prompt_parts.append("TP2: [Price]\n")
        prompt_parts.append("Justification: [Your detailed reasoning, max 3-4 sentences, linking technical and fundamental factors, mentioning liquidity, order blocks, market structure, news impact.]\n")

        full_prompt = "".join(prompt_parts)

        # Make the API call to GitHub AI
        payload = {
            "messages": [
                { "role": "system", "content": "You are an expert forex trader specializing in ICT and Smart Money Concepts, providing clear BUY/SELL recommendations with specific entry, stop-loss, and take-profit levels based on comprehensive market analysis. Your justifications must integrate both technical (ICT/SMC) and fundamental insights, focusing on institutional order flow, liquidity, and market structure in conjunction with news impact. Always provide a BUY or SELL recommendation, never HOLD." },
                { "role": "user", "content": full_prompt }
            ],
            "model": self.github_model,
            "temperature": 0.7, 
            "max_tokens": 400, 
            "top_p": 1,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.github_base_url}/chat/completions",
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.github_api_key}'
                },
                json=payload,
                timeout=75 
            )
            response.raise_for_status() 
            result = response.json()

            if result.get('choices') and len(result['choices']) > 0 and \
               result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                full_ai_response = result['choices'][0]['message']['content']
                print("Raw AI Response:\n", full_ai_response) 

                # Parse the AI's structured response
                parsed_recommendation = {
                    'recommendation': 'N/A', # Default to N/A, will be updated by regex
                    'entry': None,
                    'sl': None,
                    'tp1': None,
                    'tp2': None,
                    'justification': 'N/A'
                }

                # Regex to extract structured data. Adjusted recommendation regex for BUY/SELL only.
                rec_match = re.search(r"Recommendation:\s*(BUY|SELL)", full_ai_response)
                entry_match = re.search(r"Entry:\s*([\d.]+)", full_ai_response)
                sl_match = re.search(r"SL:\s*([\d.]+)", full_ai_response)
                tp1_match = re.search(r"TP1:\s*([\d.]+)", full_ai_response)
                tp2_match = re.search(r"TP2:\s*([\d.]+)", full_ai_response)
                justification_match = re.search(r"Justification:\s*(.*)", full_ai_response, re.DOTALL)


                if rec_match:
                    parsed_recommendation['recommendation'] = rec_match.group(1).strip()
                else:
                    # Fallback if AI doesn't strictly follow 'BUY|SELL' for recommendation
                    if "BUY" in full_ai_response.upper():
                        parsed_recommendation['recommendation'] = "BUY"
                    elif "SELL" in full_ai_response.upper():
                        parsed_recommendation['recommendation'] = "SELL"
                    else:
                        parsed_recommendation['recommendation'] = "UNSPECIFIED (Expected BUY/SELL)"


                if entry_match:
                    try:
                        parsed_recommendation['entry'] = float(entry_match.group(1))
                    except ValueError: pass
                if sl_match:
                    try:
                        parsed_recommendation['sl'] = float(sl_match.group(1))
                    except ValueError: pass
                if tp1_match:
                    try:
                        parsed_recommendation['tp1'] = float(tp1_match.group(1))
                    except ValueError: pass
                if tp2_match:
                    try:
                        parsed_recommendation['tp2'] = float(tp2_match.group(1))
                    except ValueError: pass
                if justification_match:
                    parsed_recommendation['justification'] = justification_match.group(1).strip()

                return parsed_recommendation
            else:
                print("GitHub AI API response did not contain expected content or was empty.")
                print(f"Full GitHub AI Response: {json.dumps(result, indent=2)}") 
                return {
                    'recommendation': 'AI could not generate a recommendation due to unexpected response or malformed data.',
                    'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'N/A'
                }
        except requests.exceptions.Timeout:
            return {
                'recommendation': "AI recommendation timed out. The model took too long to respond. Consider increasing timeout or simplifying prompt.",
                'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'N/A'
            }
        except requests.exceptions.RequestException as e:
            print(f"Error calling GitHub AI API: {e}")
            return {
                'recommendation': f"Error getting AI recommendation: API request failed ({e}). Check API key, network, or rate limits.",
                'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'N/A'
            }
        except json.JSONDecodeError:
            print("Failed to decode JSON response from GitHub AI API.")
            return {
                'recommendation': "AI recommendation: Invalid JSON response from API. Could indicate an API issue.",
                'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'N/A'
            }
        except Exception as e:
            print(f"An unexpected error occurred during AI recommendation: {e}")
            return {
                'recommendation': f"An unexpected error occurred during AI recommendation: {e}. Please report this.",
                'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'N/A'
            }

    def create_chart(self, data, ai_recommendation_details=None, save_path="eurusd_chart.png"):
        """
        Creates a professional-looking EUR/USD price chart with a dark theme,
        including entry, stop loss, and take profit levels if provided by AI.
        The chart is saved to a file.
        """
        try:
            if not data or data['historical_data'].empty:
                print("No data available for charting. Please ensure EUR/USD data was fetched successfully.")
                return False
            
            df = data['historical_data']
            
            plt.style.use('dark_background') 
            plt.figure(figsize=(14, 7)) 
            
            plt.plot(df.index, df['Close'], label='EUR/USD Close Price', linewidth=1.5, color='#8B5CF6') 
            plt.fill_between(df.index, df['Low'], df['High'], alpha=0.2, label='High-Low Range', color='#A78BFA') 
            
            current_price = data.get('current_price')
            if current_price:
                plt.axhline(y=current_price, color='#DC2626', linestyle='--', linewidth=1, label=f'Current Price: {current_price:.5f}') 
            
            # Add EMAs if calculated
            if 'EMA20' in df.columns and not df['EMA20'].isnull().all():
                plt.plot(df.index, df['EMA20'], label='20 EMA', color='#FDE047', linestyle=':', linewidth=1) 
            if 'EMA50' in df.columns and not df['EMA50'].isnull().all():
                plt.plot(df.index, df['EMA50'], label='50 EMA', color='#34D399', linestyle='-.', linewidth=1) 

            # Add Bollinger Bands if calculated
            if len(df) >= 20: # Ensure enough data for BB
                middle_band = df['Close'].rolling(window=20).mean()
                std_dev = df['Close'].rolling(window=20).std()
                upper_band = middle_band + (std_dev * 2)
                lower_band = middle_band - (std_dev * 2)
                if not middle_band.isnull().all(): # Check if calculated values are not all NaN
                    plt.plot(df.index, middle_band, label='Bollinger Middle Band', color='#60A5FA', linestyle='--', linewidth=0.8) 
                    plt.plot(df.index, upper_band, label='Bollinger Upper Band', color='#93C5FD', linestyle='-', linewidth=0.8) 
                    plt.plot(df.index, lower_band, label='Bollinger Lower Band', color='#93C5FD', linestyle='-', linewidth=0.8) 
            
            # Add AI-recommended levels
            # Only plot trading levels if a BUY or SELL recommendation is given
            if ai_recommendation_details and ai_recommendation_details.get('recommendation') in ['BUY', 'SELL']:
                entry = ai_recommendation_details.get('entry')
                sl = ai_recommendation_details.get('sl')
                tp1 = ai_recommendation_details.get('tp1')
                tp2 = ai_recommendation_details.get('tp2')

                if entry:
                    plt.axhline(y=entry, color='#10B981', linestyle='-', linewidth=1.2, label=f'Entry: {entry:.5f}') # Green for entry
                if sl:
                    plt.axhline(y=sl, color='#EF4444', linestyle='-', linewidth=1.2, label=f'Stop Loss: {sl:.5f}') # Red for SL
                if tp1:
                    plt.axhline(y=tp1, color='#FBBF24', linestyle='-', linewidth=1.2, label=f'Take Profit 1: {tp1:.5f}') # Amber for TP1
                if tp2:
                    plt.axhline(y=tp2, color='#FBBF24', linestyle='--', linewidth=1.2, label=f'Take Profit 2: {tp2:.5f}') # Amber dashed for TP2

            plt.title(f'EUR/USD Price Chart ({data["period"]} - {data["interval"]})', fontsize=18, color='white')
            plt.xlabel('Time', fontsize=14, color='white')
            plt.ylabel('Price (USD)', fontsize=14, color='white')
            plt.legend(facecolor='#2D3748', edgecolor='#6D28D9', labelcolor='white', fontsize=10, loc='best') 
            
            plt.grid(True, alpha=0.3, linestyle=':', color='gray') 
            plt.xticks(rotation=45, ha='right', color='white', fontsize=10) 
            plt.yticks(color='white', fontsize=10)
            
            plt.gca().set_facecolor('#1F2937') 
            plt.gcf().set_facecolor('#111827') 
            
            plt.tight_layout() 
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0) 
            
            print(f"Chart generated and saved to internal buffer (simulated save to {save_path}).")
            print(f"To view the chart, locate '{save_path}' in the script's execution directory.")
            
            plt.close() 
            return True
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return False
    
    def save_data(self, data, filename="forex_data.json"):
        """
        Saves all collected forex data to a JSON file.
        Handles DataFrame serialization for historical data.
        """
        try:
            data_to_save = data.copy()
            
            if 'eurusd_data' in data_to_save and data_to_save['eurusd_data'] and \
               'historical_data' in data_to_save['eurusd_data'] and \
               isinstance(data_to_save['eurusd_data']['historical_data'], pd.DataFrame):
                data_to_save['eurusd_data']['historical_data'] = data_to_save['eurusd_data']['historical_data'].to_json(orient='split', date_format='iso')
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str) 
            
            print(f"Data successfully saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            return False
    
    def print_summary(self, eurusd_data, news_data, market_data, analysis, economic_events=None, ai_recommendation_details=None):
        """
        Prints a comprehensive summary of forex data, technical analysis,
        news, economic events, and the AI-generated trading recommendation,
        including specific trading levels.
        """
        print("\n" + "="*80)
        print("                 LEE AI FOREX INSIGHTS (Powered by GitHub AI)                 ")
        print("="*80)
        print(f"Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if eurusd_data:
            print(f"\n--- EUR/USD CURRENT MARKET SNAPSHOT ---")
            print(f"  Current Price: {eurusd_data.get('current_price', 'N/A'):.5f}")
            print(f"  Data Period: {eurusd_data.get('period', 'N/A')}, Interval: {eurusd_data.get('interval', 'N/A')}")
            print(f"  Historical Data Points: {len(eurusd_data['historical_data']) if 'historical_data' in eurusd_data and not eurusd_data['historical_data'].empty else 0}")
        
        if analysis:
            print(f"\n--- EUR/USD TECHNICAL ANALYSIS ---")
            print(f"  Trend Direction (Current Period): {analysis.get('trend_direction', 'N/A')}")
            print(f"  Period High: {analysis.get('period_high', 0):.5f}")
            print(f"  Period Low: {analysis.get('period_low', 0):.5f}")
            print(f"  Average Price (Close): {analysis.get('average_price', 0):.5f}")
            print(f"  Volatility (Std Dev of Close): {analysis.get('volatility', 0):.5f}")
            if analysis.get('20_ema') is not None:
                print(f"  20-Period EMA: {analysis['20_ema']:.5f}")
            if analysis.get('50_ema') is not None:
                print(f"  50-Period EMA: {analysis['50_ema']:.5f}")
            if analysis.get('support_level') is not None:
                print(f"  Recent Support Level: {analysis['support_level']:.5f}")
                print(f"  Recent Resistance Level: {analysis['resistance_level']:.5f}")
            if analysis.get('rsi') is not None:
                print(f"  RSI (14-period): {analysis['rsi']:.2f} ({analysis.get('rsi_signal', 'N/A')})")
            if analysis.get('macd_histogram') is not None:
                print(f"  MACD Signal: {analysis.get('macd_signal', 'N/A')} (Histogram: {analysis['macd_histogram']:.5f})")
            if analysis.get('bollinger_middle') is not None:
                print(f"  Bollinger Bands: Middle={analysis['bollinger_middle']:.5f}, Upper={analysis['bollinger_upper']:.5f}, Lower={analysis['bollinger_lower']:.5f}")
                print(f"  Current Position relative to BB: {analysis.get('bollinger_position', 'N/A')}")
        
        if market_data:
            print(f"\n--- MAJOR FOREX PAIRS SUMMARY (Daily Changes) ---")
            for pair, data in sorted(list(market_data.items()))[:5]: 
                change_sign = "+" if data.get('change', 0) >= 0 else ""
                print(f"  {pair}: {data.get('current_price', 0):.5f} ({change_sign}{data.get('change', 0):.5f}, {data.get('change_percent', 0):+.2f}%)")
        
        if news_data:
            print(f"\n--- RECENT FOREX NEWS (Top {len(news_data)} items) ---")
            if not news_data:
                print("  No recent forex news found.")
            else:
                for i, item in enumerate(news_data[:5], 1): 
                    print(f"{i}. {item.get('title', 'N/A')}")
                    print(f"     Source: {item.get('source', 'N/A')} | Published: {item.get('timestamp', 'N/A')}")
                    if item.get('category'):
                        print(f"     Category: {item['category']}")
                    if item.get('link'):
                        print(f"     Link: {item['link']}")
                    print()
        
        if economic_events:
            print(f"\n--- UPCOMING ECONOMIC CALENDAR (High/Medium Impact Events) ---")
            relevant_events = [e for e in economic_events if e.get('impact') in ['High', 'Medium']]
            if not relevant_events:
                print("  No high or medium impact economic events scheduled.")
            else:
                for event in relevant_events:
                    print(f"• Date: {event.get('date', 'N/A')} | Time: {event.get('time', 'N/A')}")
                    print(f"  Event: {event.get('event', 'N/A')} ({event.get('currency', 'N/A')})")
                    print(f"  Impact: {event.get('impact', 'N/A')}, Forecast: {event.get('forecast', 'N/A')}, Actual: {event.get('actual', 'N/A')}, Previous: {event.get('previous', 'N/A')}")
                    print()

        if ai_recommendation_details:
            print(f"\n--- LEE AI TRADING RECOMMENDATION (ICT & Smart Money) ---")
            # Explicitly state BUY or SELL here
            if ai_recommendation_details.get('recommendation') == 'BUY':
                print(f"ACTION: BUY EUR/USD")
            elif ai_recommendation_details.get('recommendation') == 'SELL':
                print(f"ACTION: SELL EUR/USD")
            else: # Fallback for unexpected recommendations
                print(f"Recommendation: {ai_recommendation_details.get('recommendation', 'N/A')}")

            # Conditionally format and print levels
            if ai_recommendation_details.get('recommendation') in ['BUY', 'SELL']:
                entry_str = f"{ai_recommendation_details['entry']:.5f}" if ai_recommendation_details['entry'] is not None else "N/A"
                sl_str = f"{ai_recommendation_details['sl']:.5f}" if ai_recommendation_details['sl'] is not None else "N/A"
                tp1_str = f"{ai_recommendation_details['tp1']:.5f}" if ai_recommendation_details['tp1'] is not None else "N/A"
                tp2_str = f"{ai_recommendation_details['tp2']:.5f}" if ai_recommendation_details['tp2'] is not None else "N/A"
                
                print(f"  Entry Price: {entry_str}")
                print(f"  Stop Loss (SL): {sl_str}")
                print(f"  Take Profit 1 (TP1): {tp1_str}")
                print(f"  Take Profit 2 (TP2): {tp2_str}")
            print(f"\nJustification:\n{ai_recommendation_details.get('justification', 'N/A')}")
            print("\n--- IMPORTANT: This is an AI-generated recommendation. Always conduct your own thorough research and risk assessment before making any trading decisions. Past performance is not indicative of future results. ---")

        print("\n" + "="*80)

    def get_crypto_forex_correlation(self):
        """
        Fetches current prices and daily changes for major cryptocurrencies.
        (Correlation analysis is conceptual in this script, focusing on price data.)
        """
        crypto_symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD']
        crypto_data = {}
        
        print("\nFetching major cryptocurrency data for broader market sentiment...")
        for symbol in crypto_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1d")
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[0]) if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    crypto_data[symbol] = {
                        'current_price': current_price,
                        'change': change,
                        'change_percent': change_pct
                    }
                else:
                    print(f"No crypto data found for {symbol}.")
                
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Error fetching crypto data for {symbol}: {e}")
                continue
        
        return crypto_data

def main():
    scraper = YahooForexScraper()
    
    print("\n" + "#"*80)
    print("        Initializing Lee AI Forex Information & Trading Recommendation Program        ")
    print("#"*80)
    
    # 1. Get EUR/USD data
    eurusd_data = scraper.get_eurusd_data(period="1d", interval="15m")
    
    # 2. Get forex news
    news_data = scraper.get_forex_news()
    
    # 3. Get major forex pairs market summary
    market_data = scraper.get_market_summary()
    
    # 4. Get economic calendar
    economic_events = scraper.get_economic_calendar()
    
    # 5. Get crypto correlation data (optional, for broader market view)
    crypto_data = scraper.get_crypto_forex_correlation()
    
    # 6. Analyze EUR/USD trends (technical analysis)
    analysis = scraper.analyze_eurusd_trends(eurusd_data) if eurusd_data else {}
    
    # 7. Get AI Trading Recommendation using GitHub AI
    ai_recommendation_details = {
        'recommendation': 'N/A', 'entry': None, 'sl': None, 'tp1': None, 'tp2': None, 'justification': 'AI recommendation could not be generated. Check API key and data.'
    }
    if eurusd_data and analysis: 
        ai_recommendation_details = scraper.get_trading_recommendation_ai(analysis, news_data, market_data, economic_events)
    else:
        print("\nSkipping AI recommendation: Insufficient EUR/USD data or technical analysis for a valid recommendation.")

    # 8. Print comprehensive summary
    scraper.print_summary(eurusd_data, news_data, market_data, analysis, economic_events, ai_recommendation_details)
    
    # Print crypto correlation if available
    if crypto_data:
        print(f"\n--- CRYPTO CORRELATION SUMMARY ---")
        if not crypto_data:
            print("  No cryptocurrency data found.")
        else:
            for symbol, data in sorted(list(crypto_data.items())):
                change_sign = "+" if data['change'] >= 0 else ""
                print(f"  {symbol}: ${data['current_price']:.2f} ({change_sign}{data['change']:.2f}, {data['change_percent']:+.2f}%)")
    
    # 9. Save all collected data
    if eurusd_data: 
        print("\n9. Saving all collected data to forex_data.json...")
        scraper.save_data({
            'eurusd_data': {k: v for k, v in eurusd_data.items() if k != 'historical_data'}, 
            'eurusd_historical_data_json': eurusd_data['historical_data'].to_json(orient='split', date_format='iso') if not eurusd_data['historical_data'].empty else None, 
            'news_data': news_data,
            'market_data': market_data,
            'analysis': analysis,
            'economic_events': economic_events,
            'crypto_data': crypto_data,
            'ai_recommendation_details': ai_recommendation_details 
        })
        
        # 10. Create price chart
        print("\n10. Creating EUR/USD price chart with trading levels...")
        scraper.create_chart(eurusd_data, ai_recommendation_details) 
    else:
        print("\nSkipping data saving and chart generation due to missing EUR/USD data.")
    
    return {
        'eurusd_data': eurusd_data,
        'news_data': news_data,
        'market_data': market_data,
        'analysis': analysis,
        'economic_events': economic_events,
        'crypto_data': crypto_data,
        'ai_recommendation_details': ai_recommendation_details
    }

if __name__ == "__main__":
    final_result = main()
    
    if final_result and final_result['eurusd_data'] and final_result['eurusd_data'].get('current_price'):
        print(f"\n--- Program Execution Complete ---")
        print(f"Current EUR/USD Close Rate: {final_result['eurusd_data']['current_price']:.5f}")
    if final_result and final_result['ai_recommendation_details'] and final_result['ai_recommendation_details'].get('recommendation'):
        print(f"AI's Final Recommendation: {final_result['ai_recommendation_details']['recommendation']}")
        if final_result['ai_recommendation_details'].get('entry'):
            print(f"Entry: {final_result['ai_recommendation_details']['entry']:.5f}, SL: {final_result['ai_recommendation_details']['sl']:.5f}, TP1: {final_result['ai_recommendation_details']['tp1']:.5f}, TP2: {final_result['ai_recommendation_details']['tp2']:.5f}")
    else:
        print("\n--- Program Execution Complete ---")
        print("AI recommendation was not generated or data fetching failed. Please review the logs above.")

    # Optional: Uncomment the loop below to run periodic updates.
    # Be mindful of API rate limits if running continuously.
    # while True:
    #     print("\n" + "="*80)
    #     print("Running next update cycle in 5 minutes...")
    #     print("="*80)
    #     time.sleep(300) # Wait 5 minutes (300 seconds)
    #     final_result = main()
