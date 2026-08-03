import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="株価トレンドAI診断")

# --- 銘柄リスト ---
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
    "住友商事(株)": "8053.T"
}

# --- サイドバー設定 ---
st.sidebar.title("設定")
selected_name = st.sidebar.selectbox("分析対象の銘柄を選択してください", list(tickers.keys()))
ticker_symbol = tickers[selected_name]

if st.sidebar.button("🔄 データを最新に更新 (キャッシュクリア)"):
    st.cache_data.clear()

# --- 用語集 ---
with st.sidebar.expander("💡 シグナル定義・用語集"):
    st.markdown("""
    ### 📈 買いシグナル
    * **パーフェクトオーダー(買い)**
      5日・20日・60日の移動平均線が上から順に並んだ状態。強い上昇トレンドを示します。
    * **新高値ブレイクアウト**
      過去半年（125日）の最高値を終値で更新した状態。上値抵抗線の突破を示します。
    * **ゴールデンクロス**
      短期線が中長期線を下から上へ突き抜けた状態。上昇転換のサインです。
    * **グランビルの法則 (買い)**
      ①(買い転換): 20日線が下落後、横ばい/上向きに転じ、5日線が下から上に抜けた場合
      ②(押し目買い): 20日線が上向きの時、5日線が下落して20日線を下回るも、再度上昇して下から上に抜けた場合
      ③(買い乗せ): 20日線が上向きの時、5日線がいったん下落するも20日線を下抜けせずに再度上昇する場合
    * **RSI売られすぎ**
      RSIが30%を下回った状態。短期的な反発が起こりやすい水準です。

    ---
    ### 📉 売りシグナル
    * **ローソク足(包み足・陰線)**
      高値圏で前日の陽線を完全に包み込む大陰線。強い天井（下落）のサインです。
    * **ローソク足(否定陰線)**
      高値圏で前日の上昇分を完全に打ち消す陰線。上昇エネルギーの枯渇を示します。
    * **デッドクロス**
      短期線が中長期線を上から下へ突き抜けた状態。下落転換のサインです。
    * **グランビルの法則 (売り)**
      ①(売り転換): 20日線が上昇後、横ばい/下向きに転じ、5日線が上から下に抜けた場合
      ②(戻り売り): 20日線が下向きの時、5日線が上昇して20日線を上回るも、再度下落して上から下に抜けた場合
      ③(売り乗せ): 20日線が下向きの時、5日線がいったん上昇するも20日線を上に抜けずに再度下落する場合
    * **RSI買われすぎ**
      RSIが70%を上回った状態。利益確定売りが出やすく、反落に警戒が必要です。
    """)

# --- データ取得 (5年分) ---
@st.cache_data
def load_data(ticker):
    df = yfinance_download(ticker, period="5y")
    return df

def yfinance_download(ticker, period="5y"):
    df = yf.download(ticker, period=period)
    # yfinance仕様変更対応：MultiIndexの解除とSeriesの強制1次元化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close'])
    return df

# --- シグナル計算 ---
def calculate_signals(df):
    df = df.copy()
    # 移動平均
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # 高値・安値・前日比等
    df['High_125'] = df['High'].rolling(window=125).max().shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 買いシグナルの判定 ---
    buy_signals = []
    
    # 1. パーフェクトオーダー
    po_buy = (df['SMA_5'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_60'])
    po_buy_signal = po_buy & ~po_buy.shift(1).fillna(False)
    
    # 2. 新高値ブレイクアウト
    breakout_buy = df['Close'] > df['High_125']
    
    # 3. ゴールデンクロス (5日線が20日線を上抜け)
    gc_buy = (df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))
    
    # 4. グランビルの法則 (買い)
    # ①買い転換: 20日線が下落後、横ばい/上向きに転じ、5日線が下から上に抜けた場合
    sma20_slope = df['SMA_20'].diff()
    granville_buy_1 = (sma20_slope.shift(1) <= 0) & (sma20_slope >= 0) & gc_buy
    
    # ②押し目買い: 20日線上向き時、5日線が下落して20日線を下回るも、再度上昇して上抜け
    granville_buy_2 = (sma20_slope > 0) & gc_buy & (df['SMA_5'].shift(2) > df['SMA_20'].shift(2))
    
    # ③買い乗せ: 20日線上向き時、5日線が下落するも20日線を下抜けせず再上昇
    granville_buy_3 = (sma20_slope > 0) & (df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].diff() > 0) & (df['SMA_5'].diff().shift(1) < 0)
    
    granville_buy = granville_buy_1 | granville_buy_2 | granville_buy_3
    
    # 5. RSI売られすぎ
    rsi_buy = df['RSI'] < 30
    
    # --- 売りシグナルの判定 ---
    # 高値圏の判定 (終値が20日線より上)
    high_zone = df['Close'] > df['SMA_20']
    
    # 1. ローソク足 (包み足・陰線)
    # 前日陽線 (終値 > 始値)、当日陰線 (終値 < 始値) かつ 前日の実体を完全に包む
    prev_yosen = df['Prev_Close'] > df['Prev_Open']
    today_insen = df['Close'] < df['Open']
    tsutsumi_ashi = high_zone & prev_yosen & today_insen & (df['Open'] > df['Prev_Close']) & (df['Close'] < df['Prev_Open'])
    
    # 2. ローソク足 (否定陰線)
    # 前日陽線、当日陰線で、前日上昇分を打ち消して前日始値以下で引ける
    hitei_insen = high_zone & prev_yosen & today_insen & (df['Close'] <= df['Prev_Open'])
    
    # 3. デッドクロス (5日線が20日線を下抜け)
    dc_sell = (df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))
    
    # 4. グランビルの法則 (売り)
    # ①売り転換: 20日線が上昇後、横ばい/下向きに転じ、5日線が上から下に抜けた場合
    granville_sell_1 = (sma20_slope.shift(1) >= 0) & (sma20_slope <= 0) & dc_sell
    
    # ②戻り売り: 20日線下向き時、5日線が上昇して20日線を上回るも、再度下落して下抜け
    granville_sell_2 = (sma20_slope < 0) & dc_sell & (df['SMA_5'].shift(2) < df['SMA_20'].shift(2))
    
    # ③売り乗せ: 20日線下向き時、5日線が上昇するも20日線を上に抜けず再下落
    granville_sell_3 = (sma20_slope < 0) & (df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].diff() < 0) & (df['SMA_5'].diff().shift(1) > 0)
    
    granville_sell = granville_sell_1 | granville_sell_2 | granville_sell_3
    
    # 5. RSI買われすぎ
    rsi_sell = df['RSI'] > 70
    
    # シグナル列の作成
    df['Buy_PO'] = po_buy_signal
    df['Buy_Breakout'] = breakout_buy & ~breakout_buy.shift(1).fillna(False)
    df['Buy_GC'] = gc_buy
    df['Buy_Granville'] = granville_buy
    df['Buy_RSI'] = rsi_buy & ~rsi_buy.shift(1).fillna(False)
    
    df['Sell_Tsutsumi'] = tsutsumi_ashi
    df['Sell_Hitei'] = hitei_insen
    df['Sell_DC'] = dc_sell
    df['Sell_Granville'] = granville_sell
    df['Sell_RSI'] = rsi_sell & ~rsi_sell.shift(1).fillna(False)
    
    return df

# --- AI予測 ---
def predict_trend_rf(df):
    df_ai = df.copy()
    # 必要な特徴量を計算（遅行スパン等でNaNになりすぎないように注意）
    df_ai['Ret_1d'] = df_ai['Close'].pct_change()
    df_ai['Ret_5d'] = df_ai['Close'].pct_change(5)
    features = ['SMA_5', 'SMA_20', 'RSI', 'Ret_1d', 'Ret_5d']
    df_ai = df_ai.dropna(subset=features)
    
    # 1ヶ月後(20日後)のトレンド
    df_ai['Future_Ret'] = df_ai['Close'].shift(-20) / df_ai['Close'] - 1
    
    # 3クラス分類 (上昇、横ばい、下落)
    conditions = [
        (df_ai['Future_Ret'] > 0.03),
        (df_ai['Future_Ret'] < -0.03)
    ]
    choices = [1, -1]
    df_ai['Target'] = np.select(conditions, choices, default=0)
    
    df_train = df_ai.dropna(subset=['Target', 'Future_Ret'])
    
    if len(df_train) < 50:
        return "データ不足", [0,0,0]
        
    X = df_train[features]
    y = df_train['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 【修正箇所】 df ではなく df_ai を使用して最新の特徴量を取得します
    latest_features = df_ai[features].ffill().iloc[-1:]
    probs = model.predict_proba(latest_features)[0]
    
    classes = model.classes_
    prob_dict = {c: p for c, p in zip(classes, probs)}
    
    up_prob = prob_dict.get(1, 0)
    down_prob = prob_dict.get(-1, 0)
    flat_prob = prob_dict.get(0, 0)
    
    if up_prob > 0.5:
        return "上昇見込み", [up_prob, flat_prob, down_prob]
    elif down_prob > 0.5:
        return "下落警戒", [up_prob, flat_prob, down_prob]
    else:
        return "横ばい・揉み合い", [up_prob, flat_prob, down_prob]

# --- メイン画面 ---
st.title(f"📊 {selected_name} のAIトレンド診断")
df = load_data(ticker_symbol)

if df.empty:
    st.error("データの取得に失敗しました。")
else:
    df = calculate_signals(df)
    
    st.header("Step1: AIによる1ヶ月後のトレンド予測")
    pred_text, probs = predict_trend_rf(df)
    st.subheader(f"🤖 AI予測: {pred_text}")
    st.progress(probs[0], text=f"上昇確率: {probs[0]:.1%}")
    st.progress(probs[1], text=f"横ばい確率: {probs[1]:.1%}")
    st.progress(probs[2], text=f"下落確率: {probs[2]:.1%}")
    
    st.header("Step2: 直近の買いシグナル")
    # 直近5日間を抽出
    recent_df = df.iloc[-5:]
    buy_signals_found = []
    
    for date, row in recent_df.iterrows():
        d_str = date.strftime('%m/%d')
        if row['Buy_PO']: buy_signals_found.append(f"{d_str} パーフェクトオーダー(買い)")
        if row['Buy_Breakout']: buy_signals_found.append(f"{d_str} 新高値ブレイクアウト")
        if row['Buy_GC']: buy_signals_found.append(f"{d_str} ゴールデンクロス")
        if row['Buy_Granville']: buy_signals_found.append(f"{d_str} グランビルの法則(買い)")
        if row['Buy_RSI']: buy_signals_found.append(f"{d_str} RSI売られすぎ")
        
    if buy_signals_found:
        for s in reversed(buy_signals_found):
            st.success(s)
    else:
        st.info("直近5日間に点灯した買いシグナルはありません。")
        
    st.header("Step3: 直近の売りシグナル")
    sell_signals_found = []
    for date, row in recent_df.iterrows():
        d_str = date.strftime('%m/%d')
        if row['Sell_Tsutsumi']: sell_signals_found.append(f"{d_str} ローソク足(包み足・陰線)")
        if row['Sell_Hitei']: sell_signals_found.append(f"{d_str} ローソク足(否定陰線)")
        if row['Sell_DC']: sell_signals_found.append(f"{d_str} デッドクロス")
        if row['Sell_Granville']: sell_signals_found.append(f"{d_str} グランビルの法則(売り)")
        if row['Sell_RSI']: sell_signals_found.append(f"{d_str} RSI買われすぎ")
        
    if sell_signals_found:
        for s in reversed(sell_signals_found):
            st.error(s)
    else:
        st.info("直近5日間に点灯した売りシグナルはありません。")

    st.header("Step4: チャート確認")
    fig = go.Figure()
    # ローソク足
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価'))
    # 移動平均線
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name='5日線', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20日線', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='60日線', line=dict(color='green')))
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, title="株価と移動平均線")
    st.plotly_chart(fig, use_container_width=True)
