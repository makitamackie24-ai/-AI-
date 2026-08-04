import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    """過去3年分のデータを取得"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3) # 3年分
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        return df
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=['Close'])
    return df

def calculate_signals(df):
    """各種テクニカル指標と売買シグナルを計算"""
    df = df.copy()
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high_26 + low_26) / 2

    df['Senkou_Span_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    
    df['Chikou_Span'] = df['Close'].shift(-26)

    df['High_125'] = df['High'].rolling(window=125).max().shift(1)
    
    df['Buy_Signals'] = [[] for _ in range(len(df))]
    df['Sell_Signals'] = [[] for _ in range(len(df))]

    # ValueErrorを防ぐためnumpy配列で比較
    high_zone = df['Close'].values > df['SMA_20'].values
    
    for i in range(1, len(df)):
        # --- 買いシグナル ---
        # 1. ゴールデンクロス (5日線が20日線を下から上へ抜ける)
        if df['SMA_5'].iloc[i-1] <= df['SMA_20'].iloc[i-1] and df['SMA_5'].iloc[i] > df['SMA_20'].iloc[i]:
            df['Buy_Signals'].iloc[i].append("ゴールデンクロス(5日/20日)")
            
        # 2. MACD上抜け
        if df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['Signal'].iloc[i]:
             df['Buy_Signals'].iloc[i].append("MACD上抜け")
             
        sma20_trend_up = df['SMA_20'].iloc[i] > df['SMA_20'].iloc[i-5]
        sma20_trend_down = df['SMA_20'].iloc[i] < df['SMA_20'].iloc[i-5]
        
        if sma20_trend_up and df['Close'].iloc[i-1] < df['SMA_20'].iloc[i-1] and df['Close'].iloc[i] > df['SMA_20'].iloc[i]:
            df['Buy_Signals'].iloc[i].append("グランビルの法則(買い①: 買い転換)")
            
        if sma20_trend_up and df['Close'].iloc[i-1] < df['SMA_20'].iloc[i-1] and df['Close'].iloc[i] > df['SMA_20'].iloc[i] and df['Close'].iloc[i-5] > df['SMA_20'].iloc[i-5]:
            df['Buy_Signals'].iloc[i].append("グランビルの法則(買い②: 押し目買い)")
            
        if sma20_trend_up and df['Close'].iloc[i] > df['SMA_20'].iloc[i] and df['Low'].iloc[i-1] <= df['SMA_20'].iloc[i-1] * 1.01 and df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            df['Buy_Signals'].iloc[i].append("グランビルの法則(買い③: 買い乗せ)")

        if i >= 26:
            tenkan_cross = df['Tenkan'].iloc[i-1] <= df['Kijun'].iloc[i-1] and df['Tenkan'].iloc[i] > df['Kijun'].iloc[i]
            price_above_cloud = df['Close'].iloc[i] > max(df['Senkou_Span_A'].iloc[i], df['Senkou_Span_B'].iloc[i])
            chikou_above_price = df['Close'].iloc[i] > df['Close'].iloc[i-26]
            if tenkan_cross and price_above_cloud and chikou_above_price:
                df['Buy_Signals'].iloc[i].append("一目均衡表(三役好転)")

        if df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['Low'].iloc[i] > df['Low'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i-1]:
            if df['Close'].iloc[i-1] > df['Close'].iloc[i-2] and df['Low'].iloc[i-1] > df['Low'].iloc[i-2]:
                 df['Buy_Signals'].iloc[i].append("ダウ理論(上昇波)")
                 
        if df['SMA_5'].iloc[i-1] <= df['SMA_20'].iloc[i-1] or df['SMA_20'].iloc[i-1] <= df['SMA_60'].iloc[i-1]:
            if df['SMA_5'].iloc[i] > df['SMA_20'].iloc[i] > df['SMA_60'].iloc[i] and df['SMA_60'].iloc[i] > df['SMA_60'].iloc[i-1]:
                 df['Buy_Signals'].iloc[i].append("パーフェクトオーダー(買い)")
                 
        if pd.notna(df['High_125'].iloc[i]) and df['Close'].iloc[i] > df['High_125'].iloc[i] and df['Close'].iloc[i-1] <= df['High_125'].iloc[i-1]:
             df['Buy_Signals'].iloc[i].append("過去半年の新高値ブレイクアウト")

        # --- 売りシグナル ---
        if df['SMA_5'].iloc[i-1] >= df['SMA_20'].iloc[i-1] and df['SMA_5'].iloc[i] < df['SMA_20'].iloc[i]:
            df['Sell_Signals'].iloc[i].append("デッドクロス(5日/20日)")
            
        if df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['Signal'].iloc[i]:
             df['Sell_Signals'].iloc[i].append("MACD下抜け")
             
        if sma20_trend_down and df['Close'].iloc[i-1] > df['SMA_20'].iloc[i-1] and df['Close'].iloc[i] < df['SMA_20'].iloc[i]:
            df['Sell_Signals'].iloc[i].append("グランビルの法則(売り①: 売り転換)")
            
        if sma20_trend_down and df['Close'].iloc[i-1] > df['SMA_20'].iloc[i-1] and df['Close'].iloc[i] < df['SMA_20'].iloc[i] and df['Close'].iloc[i-5] < df['SMA_20'].iloc[i-5]:
             df['Sell_Signals'].iloc[i].append("グランビルの法則(売り②: 戻り売り)")
             
        if sma20_trend_down and df['Close'].iloc[i] < df['SMA_20'].iloc[i] and df['High'].iloc[i-1] >= df['SMA_20'].iloc[i-1] * 0.99 and df['Close'].iloc[i] < df['Close'].iloc[i-1]:
             df['Sell_Signals'].iloc[i].append("グランビルの法則(売り③: 売り乗せ)")

        if df['Close'].iloc[i] < df['Close'].iloc[i-1] and df['High'].iloc[i] < df['High'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i-1]:
            if df['Close'].iloc[i-1] < df['Close'].iloc[i-2] and df['High'].iloc[i-1] < df['High'].iloc[i-2]:
                df['Sell_Signals'].iloc[i].append("ダウ理論(下降波)")
                
        if high_zone[i]:
            if df['Close'].iloc[i-1] > df['Open'].iloc[i-1]: 
                if df['Open'].iloc[i] > df['Close'].iloc[i-1] and df['Close'].iloc[i] < df['Open'].iloc[i-1]:
                    df['Sell_Signals'].iloc[i].append("ローソク足(包み足・陰線)")
            
            if df['Close'].iloc[i-1] > df['Open'].iloc[i-1]: 
                if df['Open'].iloc[i] <= df['Close'].iloc[i-1] and df['Close'].iloc[i] < df['Open'].iloc[i-1]:
                    if "ローソク足(包み足・陰線)" not in df['Sell_Signals'].iloc[i]:
                        df['Sell_Signals'].iloc[i].append("ローソク足(否定陰線)")

    return df

def calculate_signal_performance(df):
    """過去3年のシグナルごとの平均変動率とスコアを計算"""
    buy_performance = {}
    sell_performance = {}
    current_date = df.index[-1]
    
    # シグナルの重み付け設定
    weights_buy = {
        "グランビルの法則(買い①: 買い転換)": [0, 0, 0, 0.1, 0.2, 0.3, 0.4], 
        "グランビルの法則(買い②: 押し目買い)": [0, 0, 0, 0.1, 0.2, 0.3, 0.4], 
        "グランビルの法則(買い③: 買い乗せ)": [0, 0, 0, 0.2, 0.4, 0.4, 0],   
        "ゴールデンクロス(5日/20日)": [0, 0, 0, 0.1, 0.2, 0.3, 0.4],
        "MACD上抜け": [0, 0, 0, 0.1, 0.2, 0.3, 0.4],
        "一目均衡表(三役好転)": [0, 0, 0, 0.1, 0.2, 0.3, 0.4],
        "ダウ理論(上昇波)": [0, 0, 0, 0.6, 0.4, 0, 0],                     
        "パーフェクトオーダー(買い)": [0, 0, 0, 0.1, 0.2, 0.3, 0.4],
        "過去半年の新高値ブレイクアウト": [0, 0, 0, 0.1, 0.2, 0.3, 0.4]
    }
    
    weights_sell = {
        "グランビルの法則(売り①: 売り転換)": [0, 0, 0, 1.0, 0, 0, 0], 
        "グランビルの法則(売り②: 戻り売り)": [0, 0, 0, 1.0, 0, 0, 0],
        "グランビルの法則(売り③: 売り乗せ)": [0.2, 0.3, 0.5, 0, 0, 0, 0], 
        "デッドクロス(5日/20日)": [0, 0, 0, 1.0, 0, 0, 0],
        "MACD下抜け": [0, 0, 0, 1.0, 0, 0, 0],
        "ダウ理論(下降波)": [0.4, 0.3, 0.3, 0, 0, 0, 0],                 
        "ローソク足(包み足・陰線)": [0.4, 0.3, 0.3, 0, 0, 0, 0],
        "ローソク足(否定陰線)": [0.4, 0.3, 0.3, 0, 0, 0, 0]
    }

    for i in range(len(df)):
        buy_sigs = df['Buy_Signals'].iloc[i]
        sell_sigs = df['Sell_Signals'].iloc[i]
        
        date = df.index[i]
        close_price = df['Close'].iloc[i]
        
        days_diff = (current_date - date).days
        time_weight = max(0.2, 1.0 - (days_diff / 1095) * 0.8)
        
        future_prices = []
        for offset in [1, 2, 3, 5, 10, 15, 20]:
            if i + offset < len(df):
                ret = (df['Close'].iloc[i + offset] - close_price) / close_price * 100
                future_prices.append(ret)
            else:
                future_prices.append(np.nan)
                
        for sig in buy_sigs:
            if sig not in buy_performance: buy_performance[sig] = []
            buy_performance[sig].append({'returns': future_prices, 'weight': time_weight})
            
        for sig in sell_sigs:
            if sig not in sell_performance: sell_performance[sig] = []
            sell_performance[sig].append({'returns': future_prices, 'weight': time_weight})

    def aggregate_stats(performance_dict, weight_dict, is_buy=True):
        stats = []
        for sig, data_list in performance_dict.items():
            count = len(data_list)
            
            avg_returns = []
            for j in range(7):
                valid_returns = [d['returns'][j] for d in data_list if not np.isnan(d['returns'][j])]
                avg = np.mean(valid_returns) if valid_returns else np.nan
                avg_returns.append(avg)
                
            score = 0
            period_weights = weight_dict.get(sig, [0,0,0, 0.25, 0.25, 0.25, 0.25])
            total_weighted_score = 0
            weight_sum = 0
            
            for j in range(7):
                if period_weights[j] > 0:
                    valid_data = [d for d in data_list if not np.isnan(d['returns'][j])]
                    if valid_data:
                        weighted_ret_sum = sum(d['returns'][j] * d['weight'] for d in valid_data)
                        total_weight = sum(d['weight'] for d in valid_data)
                        weighted_avg_ret = weighted_ret_sum / total_weight
                        
                        eval_ret = weighted_avg_ret if is_buy else -weighted_avg_ret
                        sub_score = min(max(50 + (eval_ret * 10), 0), 100)
                        
                        total_weighted_score += sub_score * period_weights[j]
                        weight_sum += period_weights[j]
            
            final_score = int(total_weighted_score / weight_sum) if weight_sum > 0 else 50
            if count < 3: final_score = int(final_score * 0.8)

            stat = {
                'シグナル': sig,
                '点灯回数': count,
                '精度スコア': final_score,
            }
            if is_buy:
                stat['1週間後'] = avg_returns[3]
                stat['2週間後'] = avg_returns[4]
                stat['3週間後'] = avg_returns[5]
                stat['4週間後'] = avg_returns[6]
            else:
                stat['1日後'] = avg_returns[0]
                stat['2日後'] = avg_returns[1]
                stat['3日後'] = avg_returns[2]
                stat['1週間後'] = avg_returns[3]
            
            stats.append(stat)
        
        df_stats = pd.DataFrame(stats)
        if not df_stats.empty:
            df_stats = df_stats.sort_values('精度スコア', ascending=False).reset_index(drop=True)
        return df_stats

    df_buy_stats = aggregate_stats(buy_performance, weights_buy, is_buy=True)
    df_sell_stats = aggregate_stats(sell_performance, weights_sell, is_buy=False)
    
    return df_buy_stats, df_sell_stats

def predict_trend_rf(df):
    """ランダムフォレストによる1ヶ月後のトレンド予測 (3クラス)"""
    features = ['SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'Signal', 'RSI']
    
    df['Future_Close'] = df['Close'].shift(-20)
    df['Return_20d'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    def classify_trend(ret):
        if pd.isna(ret): return np.nan
        if ret >= 0.03: return 2 # Up
        elif ret <= -0.03: return 0 # Down
        else: return 1 # Neutral
        
    df['Target'] = df['Return_20d'].apply(classify_trend)
    
    model_df = df[features + ['Target']].copy()
    model_df = model_df.dropna()
    
    if len(model_df) < 100:
        return None, None
        
    X = model_df[features]
    y = model_df['Target']
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    # Pandas最新仕様に合わせた欠損値補完処理（エラー対策）
    latest_features = df[features].ffill().iloc[-1:]
    prediction = rf.predict(latest_features)[0]
    probabilities = rf.predict_proba(latest_features)[0]
    
    classes = ["下降 (Down)", "もみ合い (Neutral)", "上昇 (Up)"]
    result_text = classes[int(prediction)]
    
    probs_dict = {
        "下降": probabilities[0] if len(probabilities) > 0 else 0,
        "もみ合い": probabilities[1] if len(probabilities) > 1 else 0,
        "上昇": probabilities[2] if len(probabilities) > 2 else 0
    }
    
    return result_text, probs_dict

def create_chart(df):
    """Plotlyによるインタラクティブなローソク足チャート (3年分)"""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='ローソク足',
                increasing_line_color='#ef5350', decreasing_line_color='#26a69a'))

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name='5日線', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20日線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='60日線', line=dict(color='green', width=1)))

    buy_dates_dow, buy_prices_dow, buy_texts_dow = [], [], []
    buy_dates_other, buy_prices_other, buy_texts_other = [], [], []
    sell_dates_dow, sell_prices_dow, sell_texts_dow = [], [], []
    sell_dates_other, sell_prices_other, sell_texts_other = [], [], []

    for i in range(len(df)):
        if df['Buy_Signals'].iloc[i]:
            for sig in df['Buy_Signals'].iloc[i]:
                if "ダウ理論" in sig:
                    buy_dates_dow.append(df.index[i])
                    buy_prices_dow.append(df['Low'].iloc[i] * 0.96) 
                    buy_texts_dow.append(sig)
                else:
                    buy_dates_other.append(df.index[i])
                    buy_prices_other.append(df['Low'].iloc[i] * 0.98)
                    buy_texts_other.append(sig)
                    
        if df['Sell_Signals'].iloc[i]:
            for sig in df['Sell_Signals'].iloc[i]:
                if "ダウ理論" in sig:
                    sell_dates_dow.append(df.index[i])
                    sell_prices_dow.append(df['High'].iloc[i] * 1.04) 
                    sell_texts_dow.append(sig)
                else:
                    sell_dates_other.append(df.index[i])
                    sell_prices_other.append(df['High'].iloc[i] * 1.02)
                    sell_texts_other.append(sig)

    fig.add_trace(go.Scatter(x=buy_dates_other, y=buy_prices_other, mode='markers',
                             marker=dict(symbol='triangle-up', color='cyan', size=10, line=dict(width=1, color='DarkSlateGrey')),
                             name='買いシグナル', text=buy_texts_other, hoverinfo='text+x'))
                             
    fig.add_trace(go.Scatter(x=buy_dates_dow, y=buy_prices_dow, mode='markers',
                             marker=dict(symbol='star', color='gold', size=12, line=dict(width=1, color='DarkSlateGrey')),
                             name='ダウ理論(買い)', text=buy_texts_dow, hoverinfo='text+x'))

    fig.add_trace(go.Scatter(x=sell_dates_other, y=sell_prices_other, mode='markers',
                             marker=dict(symbol='triangle-down', color='magenta', size=10, line=dict(width=1, color='DarkSlateGrey')),
                             name='売りシグナル', text=sell_texts_other, hoverinfo='text+x'))
                             
    fig.add_trace(go.Scatter(x=sell_dates_dow, y=sell_prices_dow, mode='markers',
                             marker=dict(symbol='star', color='purple', size=12, line=dict(width=1, color='white')),
                             name='ダウ理論(売り)', text=sell_texts_dow, hoverinfo='text+x'))

    fig.update_layout(
        title="株価チャート (過去3年)",
        yaxis_title='株価',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def analyze_and_display(ticker_symbol):
    """1銘柄ごとの分析とUI表示を行う"""
    df = fetch_data(ticker_symbol)
    
    if df.empty:
        st.error(f"データが取得できませんでした: {ticker_symbol}")
        return

    df = calculate_signals(df)
    df_buy_stats, df_sell_stats = calculate_signal_performance(df)
    
    score_dict_buy = dict(zip(df_buy_stats['シグナル'], df_buy_stats['精度スコア'])) if not df_buy_stats.empty else {}
    score_dict_sell = dict(zip(df_sell_stats['シグナル'], df_sell_stats['精度スコア'])) if not df_sell_stats.empty else {}

    latest_date = df.index[-1].strftime("%Y-%m-%d")
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    diff = latest_close - prev_close
    diff_pct = (diff / prev_close) * 100

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric(label=f"最新終値 ({latest_date})", 
                  value=f"{latest_close:,.1f}", 
                  delta=f"{diff:,.1f} ({diff_pct:,.2f}%)")

    recent_5d = df[['Open', 'High', 'Low', 'Close']].tail(5).sort_index(ascending=False)
    recent_5d.index = recent_5d.index.strftime('%Y-%m-%d')
    recent_5d = recent_5d.round(1)
    
    with st.expander("📊 直近5日間の四本値 (始値・高値・安値・終値)"):
        st.dataframe(recent_5d.T, use_container_width=True)

    recent_buy_signals = {}
    recent_sell_signals = {}
    
    for i in range(1, 6):
        if len(df) >= i:
            idx = -i
            date_str = df.index[idx].strftime('%m/%d')
            
            for sig in df['Buy_Signals'].iloc[idx]:
                if sig not in recent_buy_signals:
                    recent_buy_signals[sig] = []
                recent_buy_signals[sig].append(date_str)
                
            for sig in df['Sell_Signals'].iloc[idx]:
                if sig not in recent_sell_signals:
                    recent_sell_signals[sig] = []
                recent_sell_signals[sig].append(date_str)

    col_sig1, col_sig2 = st.columns(2)
    with col_sig1:
        st.markdown("### 🟢 Step2: 直近の買いシグナル")
        st.caption("※直近5営業日以内に点灯したものを表示")
        if recent_buy_signals:
            for sig, dates in recent_buy_signals.items():
                score = score_dict_buy.get(sig, "N/A")
                dates_str = ", ".join(dates)
                st.success(f"✔️ **{sig}** (精度: {score}点)\n\n点灯日: {dates_str}")
        else:
            st.write("直近5日間で点灯した買いシグナルはありません。")

    with col_sig2:
        st.markdown("### 🔴 Step3: 直近の売りシグナル")
        st.caption("※直近5営業日以内に点灯したものを表示")
        if recent_sell_signals:
            for sig, dates in recent_sell_signals.items():
                score = score_dict_sell.get(sig, "N/A")
                dates_str = ", ".join(dates)
                st.error(f"⚠️ **{sig}** (精度: {score}点)\n\n点灯日: {dates_str}")
        else:
            st.write("直近5日間で点灯した売りシグナルはありません。")

    with st.expander("📈 過去3年間のシグナル実績（平均変動割合）"):
        col_b_stat, col_s_stat = st.columns(2)
        with col_b_stat:
            st.markdown("**■ 買いシグナル実績**")
            if not df_buy_stats.empty:
                st.dataframe(df_buy_stats.style.format({
                    '1週間後': '{:.2f}%', '2週間後': '{:.2f}%', '3週間後': '{:.2f}%', '4週間後': '{:.2f}%'
                }, na_rep='-'), use_container_width=True)
            else:
                st.write("データがありません")
                
        with col_s_stat:
            st.markdown("**■ 売りシグナル実績**")
            if not df_sell_stats.empty:
                 st.dataframe(df_sell_stats.style.format({
                    '1日後': '{:.2f}%', '2日後': '{:.2f}%', '3日後': '{:.2f}%', '1週間後': '{:.2f}%'
                }, na_rep='-'), use_container_width=True)
            else:
                st.write("データがありません")

    st.markdown("### 🤖 Step4: AIによる1ヶ月後のトレンド予測")
    prediction_text, probs = predict_trend_rf(df)
    
    if prediction_text:
        st.markdown(f"**予測結果:** 【 {prediction_text} 】")
        st.write("各トレンドの確率:")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.caption(f"上昇: {probs['上昇']*100:.1f}%")
            st.progress(float(probs['上昇']))
        with col_p2:
            st.caption(f"もみ合い: {probs['もみ合い']*100:.1f}%")
            st.progress(float(probs['もみ合い']))
        with col_p3:
             st.caption(f"下降: {probs['下降']*100:.1f}%")
             st.progress(float(probs['下降']))
    else:
        st.write("データ不足のためAI予測を実行できません。")

    with st.expander("📉 チャートを表示 (過去3年間・シグナルマーカー付き)"):
        fig = create_chart(df)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="多角的シグナル検知AIツール", layout="wide")
    st.title("多角的シグナル検知AIツール(完全版)")
    
    # ご指定いただいた21銘柄のリスト（JX金属を削除済み、インデント修正済み）
    tickers = {
        "NTT(株)": "9432.T",
        "ソフトバンク(株)": "9434.T",
        "日本製鉄(株)": "5401.T",
        "東京電力ホールディングス(株)": "9501.T",
        "日産自動車(株)": "7201.T",
        "(株)サンリオ": "8136.T",
        "ホンダ": "7267.T",
        "伊藤忠商事(株)": "8001.T",
        "楽天グループ(株)": "4755.T",
        "野村ホールディングス(株)": "8604.T",
        "住友電気工業(株)": "5802.T",
        "(株)三菱ＵＦＪフィナンシャル・グループ": "8306.T",
        "ソニーグループ(株)": "6758.T",
        "トヨタ自動車(株)": "7203.T",
        "(株)ＳＵＭＣＯ": "3436.T",
        "三菱重工業(株)": "7011.T",
        "三菱自動車(株)": "7211.T",
        "川崎重工業(株)": "7012.T",
        "ＬＩＮＥヤフー(株)": "4689.T",
        "(株)ＩＨＩ": "7013.T",
        "塩野義製薬": "4507.T",
        "いすゞ自動車": "7202.T",
        "住友商事(株)": "8053.T"
    }

    st.sidebar.header("設定")
    selected_names = st.sidebar.multiselect("対象銘柄を選択", list(tickers.keys()), default=list(tickers.keys()))

    if st.sidebar.button("総合診断を実行", type="primary"):
        if not selected_names:
            st.warning("銘柄を選択してください。")
            return

        progress_text = "データを取得・解析中..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, name in enumerate(selected_names):
            ticker_symbol = tickers[name]
            
            progress = (i) / len(selected_names)
            my_bar.progress(progress, text=f"解析中: {name} ({i+1}/{len(selected_names)})")
            
            st.markdown(f"## {name} ({ticker_symbol})")
            analyze_and_display(ticker_symbol)
            st.markdown("---")
            
        my_bar.progress(1.0, text="解析完了！")

# --- サイドバー設定 ---
    if st.sidebar.button("🔄 キャッシュクリア"):
        st.cache_data.clear()

# --- 備忘録（シグナル定義・数値基準） ---
    with st.sidebar.expander("📖 シグナル判断の数値基準"):
        st.markdown("""
        ### 📈 買いシグナル
        * **パーフェクトオーダー(買い)**
          `5日線 > 20日線` かつ `20日線 > 60日線` が成立した最初の日。
        * **新高値ブレイクアウト**
          `当日終値 > 過去125日間の最高値` を上抜けた最初の日。
        * **ゴールデンクロス**
          当日の `5日線 > 20日線` かつ 前日の `5日線 <= 20日線`。
        * **グランビルの法則 (買い)**
          ①(転換): 20日線の傾きが `0以下→0以上` になり、GC発生。
          ②(押し目): 20日線が `上向き` で、直近で5日線が20日線を下回り、再度GC発生。
          ③(買い乗せ): 20日線が `上向き` かつ `5日線 > 20日線` で、5日線の傾きが `マイナス→プラス` に転換。
        * **MACD上抜け（ゴールデンクロス）**
 条件: 当日のMACD線 > 当日のシグナル線 かつ 前日のMACD線 <= 前日のシグナル線。
 意味: MACD線がシグナル線を下から上へ突き抜けた状態です。直近の価格モメンタムが上向きに変化したことを示し、トレンド転換や上昇の初期段階を捉える買いサインとして機能します。する強い買いサインとして機能します。
        * **RSI売られすぎ**
          14日間のRSIが `30%未満` になった最初の日。
    
        ---
        ### 📉 売りシグナル
        *(※ローソク足シグナルは `当日終値 > 20日線` の高値圏でのみ判定)*
        * **ローソク足(包み足・陰線)**
          前日陽線、当日陰線で、`当日始値 > 前日終値` かつ `当日終値 < 前日始値`（前日実体を完全に包む）。
        * **ローソク足(否定陰線)**
          前日陽線、当日陰線で、`当日終値 <= 前日始値`（前日上昇分を完全に打ち消す）。
        * **デッドクロス**
          当日の `5日線 < 20日線` かつ 前日の `5日線 >= 20日線`。
        * **グランビルの法則 (売り)**
          ①(転換): 20日線の傾きが `0以上→0以下` になり、DC発生。
          ②(戻り売り): 20日線が `下向き` で、直近で5日線が20日線を上回り、再度DC発生。
          ③(売り乗せ): 20日線が `下向き` かつ `5日線 < 20日線` で、5日線の傾きが `プラス→マイナス` に転換。
        * **MACD下抜け（デッドクロス）**
 条件: 当日のMACD線 < 当日のシグナル線 かつ 前日のMACD線 >= 前日のシグナル線。
 意味: MACD線がシグナル線を上から下へ突き抜けた状態です。直近の価格モメンタムが下向きに変化したことを示し、上昇トレンドの終わりや下落の始まりを警戒する売りサインとなります。強い売りサインとして警戒されます。
        * **RSI買われすぎ**
          14日間のRSIが `70%超` になった最初の日。
        """) 

if __name__ == "__main__":
    main()
