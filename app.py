import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ページ設定 ---
st.set_page_config(
    page_title="AIテクニカル総合診断ツール",
    layout="wide"
)

st.title("AI & マルチテクニカル 総合診断ツール")
st.write("過去5年分のデータから、グランビルの法則、ダウ理論、一目均衡表などの主要テクニカル指標と、AI（ランダムフォレスト500本）を用いた1ヶ月後のトレンド予測を統合して表示します。")

# --- Step0: 対象銘柄リスト ---
TARGET_STOCKS = {
    "9432.T": "NTT",
    "9434.T": "ソフトバンク",
    "8604.T": "野村ホールディングス",
    "5401.T": "日本製鉄",
    "9501.T": "東京電力HD",
    "7201.T": "日産自動車",
    "8136.T": "サンリオ",
    "7267.T": "ホンダ",
    "8001.T": "伊藤忠商事",
    "4755.T": "楽天グループ"
}

# --- 指標計算関数 ---
def add_technical_indicators(df):
    # 移動平均線 (5日, 20日)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 一目均衡表
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high9 + low9) / 2 # 転換線
    
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high26 + low26) / 2 # 基準線
    
    df['Senkou1'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(25) # 先行スパン1 (雲)
    
    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    df['Senkou2'] = ((high52 + low52) / 2).shift(25) # 先行スパン2 (雲)
    
    df['Chikou'] = df['Close'].shift(-25) # 遅行スパン (未来へシフト。判定時は過去のローソク足と比較)
    
    # RSI (AI特徴量用)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# --- ダウ理論判定 ---
def check_dow_theory(df):
    # 直近60日間のスイングハイ・スイングローを抽出
    period_df = df.tail(60).copy()
    
    # 5日間のローカル最大・最小を探す
    period_df['Local_Max'] = period_df['High'][(period_df['High'] == period_df['High'].rolling(5, center=True).max())]
    period_df['Local_Min'] = period_df['Low'][(period_df['Low'] == period_df['Low'].rolling(5, center=True).min())]
    
    highs = period_df['Local_Max'].dropna().values
    lows = period_df['Local_Min'].dropna().values
    
    up_trend = False
    down_trend = False
    trigger_date = None
    
    # 高値と安値がそれぞれ2つ以上ある場合のみ判定
    if len(highs) >= 2 and len(lows) >= 2:
        h1, h2 = highs[-2], highs[-1] # 最後から2番目と最後の高値
        l1, l2 = lows[-2], lows[-1]   # 最後から2番目と最後の安値
        
        last_h_idx = period_df['Local_Max'].dropna().index[-1]
        last_l_idx = period_df['Local_Min'].dropna().index[-1]
        trigger_date = pd.to_datetime(max(last_h_idx, last_l_idx)).strftime('%Y-%m-%d')
        
        if h2 > h1 and l2 > l1:
            up_trend = True
        elif h2 < h1 and l2 < l1:
            down_trend = True
            
    return up_trend, down_trend, trigger_date

# --- 一目均衡表判定 ---
def check_ichimoku(df):
    if len(df) < 52:
        return False, False, None
        
    curr = df.iloc[-1]
    past_26 = df.iloc[-26] # 遅行スパンとの比較対象 (26日前のローソク足)
    
    cloud_top = max(curr['Senkou1'], curr['Senkou2'])
    cloud_bottom = min(curr['Senkou1'], curr['Senkou2'])
    
    # 三役好転
    cond1_up = curr['Tenkan'] > curr['Kijun'] # 転換線が基準線を上回る
    cond2_up = curr['Close'] > cloud_top      # ローソク足が雲を上抜ける
    cond3_up = curr['Close'] > past_26['Close'] # 今の終値(遅行スパン)が26日前の株価を上回る
    up_turn = bool(cond1_up and cond2_up and cond3_up)
    
    # 三役逆転
    cond1_dn = curr['Tenkan'] < curr['Kijun']
    cond2_dn = curr['Close'] < cloud_bottom
    cond3_dn = curr['Close'] < past_26['Close']
    dn_turn = bool(cond1_dn and cond2_dn and cond3_dn)
    
    trigger_date = pd.to_datetime(curr.name).strftime('%Y-%m-%d')
    
    return up_turn, dn_turn, trigger_date

# --- AI予測モデル学習・推論 ---
def run_ai_prediction(df):
    # 未来20営業日後の価格変化率を計算 (+5%で上昇トレンド、-5%で下落トレンドと定義)
    df['Future_Return'] = df['Close'].shift(-20) / df['Close'] - 1.0
    
    # 目的変数クラス作成
    df['Target_Up'] = (df['Future_Return'] >= 0.05).astype(int)
    df['Target_Down'] = (df['Future_Return'] <= -0.05).astype(int)
    
    features = ['Close', 'SMA_5', 'SMA_20', 'MACD', 'MACD_Signal', 'RSI']
    
    # NaNを含む行（直近20日分や移動平均計算初期）を除外して学習データ作成
    train_df = df.dropna(subset=features + ['Future_Return']).copy()
    
    if len(train_df) < 100:
        return 0.0, 0.0
        
    X = train_df[features]
    
    # 上昇予測モデル (木500本)
    y_up = train_df['Target_Up']
    if len(y_up.unique()) > 1:
        model_up = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
        model_up.fit(X, y_up)
        prob_up = model_up.predict_proba(df[features].iloc[-1:])[:, 1][0]
    else:
        prob_up = 0.0
        
    # 下落予測モデル (木500本)
    y_down = train_df['Target_Down']
    if len(y_down.unique()) > 1:
        model_down = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
        model_down.fit(X, y_down)
        prob_down = model_down.predict_proba(df[features].iloc[-1:])[:, 1][0]
    else:
        prob_down = 0.0
        
    return prob_up * 100, prob_down * 100

# --- メイン解析処理 ---
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_all_stocks():
    results = []
    
    # 過去5年分のデータ取得 (Step0)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)
    tickers = list(TARGET_STOCKS.keys())
    
    df_all = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        name = TARGET_STOCKS[ticker]
        
        try:
            df = df_all[ticker].copy() if len(tickers) > 1 else df_all.copy()
            df.dropna(how='all', inplace=True)
            
            if len(df) < 200: # データ不足の場合はスキップ
                continue
                
            # マルチインデックスの解消
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = add_technical_indicators(df)
            
            def get_val(series, idx=-1):
                val = series.iloc[idx]
                return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

            # --- Step1: トレンド判定 ---
            curr_close = get_val(df['Close'])
            curr_sma20 = get_val(df['SMA_20'])
            past5_sma20 = get_val(df['SMA_20'], -5)
            
            # 20日線の傾き（5日間での変化率）
            sma20_slope = (curr_sma20 - past5_sma20) / past5_sma20 * 100
            
            if sma20_slope > 0.3 and curr_close > curr_sma20:
                current_trend = "上昇トレンド ↗️"
            elif sma20_slope < -0.3 and curr_close < curr_sma20:
                current_trend = "下落トレンド ↘️"
            else:
                current_trend = "ボックストレンド ➡️"
                
            # --- Step2 & 3: テクニカルサイン判定 ---
            # グランビルの法則 (直近1ヶ月間（20営業日）でクロスが発生しているか判定)
            granville_buy = False
            granville_buy_date = None
            granville_sell = False
            granville_sell_date = None
            for j in range(1, 21): # 検出期間を直近20日間に拡大
                c0, c1 = get_val(df['Close'], -j), get_val(df['Close'], -(j+1))
                s0, s1 = get_val(df['SMA_20'], -j), get_val(df['SMA_20'], -(j+1))
                
                curr_date_str = pd.to_datetime(df.index[-j]).strftime('%Y-%m-%d')
                
                # 買い: 20日線が上向きor横ばいで、株価が下から上へクロス
                if s0 >= s1 and c0 > s0 and c1 <= s1:
                    if not granville_buy:
                        granville_buy = True
                        granville_buy_date = curr_date_str
                # 売り: 20日線が下向きor横ばいで、株価が上から下へクロス
                if s0 <= s1 and c0 < s0 and c1 >= s1:
                    if not granville_sell:
                        granville_sell = True
                        granville_sell_date = curr_date_str
                    
            latest_date_str = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d')
            
            # 移動平均線クロス (5日線と20日線)
            sma5_0, sma5_1 = get_val(df['SMA_5']), get_val(df['SMA_5'], -2)
            sma20_0, sma20_1 = curr_sma20, get_val(df['SMA_20'], -2)
            gc = bool(sma5_0 > sma20_0 and sma5_1 <= sma20_1)
            dc = bool(sma5_0 < sma20_0 and sma5_1 >= sma20_1)
            
            # MACDクロス
            macd_0, macd_1 = get_val(df['MACD']), get_val(df['MACD'], -2)
            sig_0, sig_1 = get_val(df['MACD_Signal']), get_val(df['MACD_Signal'], -2)
            macd_buy = bool(macd_0 > sig_0 and macd_1 <= sig_1)
            macd_sell = bool(macd_0 < sig_0 and macd_1 >= sig_1)
            
            # ダウ理論
            dow_up, dow_down, dow_date = check_dow_theory(df)
            
            # 一目均衡表
            ichimoku_up, ichimoku_down, ichimoku_date = check_ichimoku(df)
            
            # AI予測 (ボックストレンドの場合に確率を表示)
            prob_up, prob_down = run_ai_prediction(df)
            
            results.append({
                "name": name,
                "ticker": ticker,
                "price": curr_close,
                "trend": current_trend,
                "is_box": "ボックストレンド" in current_trend,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "signals_buy": {
                    "グランビルの法則(買い)": {"active": granville_buy, "date": granville_buy_date},
                    "ゴールデンクロス(5日/20日)": {"active": gc, "date": latest_date_str},
                    "MACD上抜け": {"active": macd_buy, "date": latest_date_str},
                    "ダウ理論(上昇波)": {"active": dow_up, "date": dow_date},
                    "一目均衡表(三役好転)": {"active": ichimoku_up, "date": ichimoku_date}
                },
                "signals_sell": {
                    "グランビルの法則(売り)": {"active": granville_sell, "date": granville_sell_date},
                    "デッドクロス(5日/20日)": {"active": dc, "date": latest_date_str},
                    "MACD下抜け": {"active": macd_sell, "date": latest_date_str},
                    "ダウ理論(下落波)": {"active": dow_down, "date": dow_date},
                    "一目均衡表(三役逆転)": {"active": ichimoku_down, "date": ichimoku_date}
                },
                "chart_data": df.tail(150).copy() # チャート描画用に直近約半年分のデータを保存
            })
            
        except Exception as e:
            st.error(f"{name} のデータ処理中にエラーが発生しました: {e}")
            
        progress_bar.progress((i + 1) / len(tickers))
        
    progress_bar.empty()
    return results

# --- 画面描画 ---
if st.button("全10銘柄の総合診断を実行 (過去5年分データ取得・AI学習)", type="primary"):
    with st.spinner("過去5年分のデータを取得し、AI（ランダムフォレスト500本）と各種テクニカル指標の解析を行っています..."):
        results = analyze_all_stocks()
        st.session_state['results'] = results

if 'results' in st.session_state:
    st.divider()
    
    for res in st.session_state['results']:
        with st.container(border=True):
            col_title, col_trend, col_ai = st.columns([1, 1, 2])
            
            with col_title:
                st.markdown(f"### {res['name']}")
                st.caption(f"コード: {res['ticker']} | 直近終値: ¥{res['price']:,.0f}")
                
            with col_trend:
                st.markdown("##### Step1: 現在のトレンド")
                # トレンドによって色を変える
                trend_color = "green" if "上昇" in res['trend'] else "red" if "下落" in res['trend'] else "gray"
                st.markdown(f"<h4 style='color: {trend_color};'>{res['trend']}</h4>", unsafe_allow_html=True)
                
            with col_ai:
                if res['is_box']:
                    st.markdown("##### 🤖 AI予測 (現在ボックス圏のため)")
                    st.markdown(f"<span style='color: #1976D2; font-weight: bold;'>1ヶ月後に上昇トレンドに乗る確率: {res['prob_up']:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #D32F2F; font-weight: bold;'>1ヶ月後に下落トレンドに乗る確率: {res['prob_down']:.1f}%</span>", unsafe_allow_html=True)
                else:
                    st.markdown("##### 🤖 AI予測")
                    st.write("※現在ボックストレンドではないため、トレンド転換確率の予測はスキップしています。")

            st.markdown("---")
            
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                st.markdown("#### 🔵 Step2: 買いタイミングのシグナル")
                buy_count = 0
                for sig_name, sig_data in res['signals_buy'].items():
                    if sig_data['active']:
                        date_str = f" [{sig_data['date']}]" if sig_data['date'] else ""
                        st.markdown(f"- ✅ **{sig_name}** 点灯 {date_str}")
                        buy_count += 1
                if buy_count == 0:
                    st.write("現在、点灯している買いシグナルはありません。")
                    
            with col_sell:
                st.markdown("#### 🔴 Step3: 売りタイミングのシグナル")
                sell_count = 0
                for sig_name, sig_data in res['signals_sell'].items():
                    if sig_data['active']:
                        date_str = f" [{sig_data['date']}]" if sig_data['date'] else ""
                        st.markdown(f"- ⚠️ **{sig_name}** 点灯 {date_str}")
                        sell_count += 1
                if sell_count == 0:
                    st.write("現在、点灯している売りシグナルはありません。")

            # --- チャート表示セクション ---
            with st.expander("📊 ローソク足チャートを表示 (直近約半年分)"):
                df_c = res['chart_data']
                
                # 2行1列のサブプロット作成 (上: ローソク足, 下: 出来高)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # ローソク足
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='株価'), row=1, col=1)
                
                # 移動平均線 (5日線・20日線)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_5'], line=dict(color='orange', width=1.5), name='5日線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_20'], line=dict(color='blue', width=1.5), name='20日線'), row=1, col=1)
                
                # 出来高 (陽線は緑系、陰線は赤系で色分け)
                colors = ['#ef5350' if row['Close'] < row['Open'] else '#26a69a' for index, row in df_c.iterrows()]
                fig.add_trace(go.Bar(x=df_c.index, y=df_c['Volume'], marker_color=colors, name='出来高'), row=2, col=1)
                
                # レイアウト調整
                fig.update_layout(
                    height=500, 
                    margin=dict(l=0, r=0, t=30, b=0), 
                    xaxis_rangeslider_visible=False, # ローソク足標準のレンジスライダーを非表示
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # 土日などの休場日の空白を詰める処理
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                
                st.plotly_chart(fig, use_container_width=True)
