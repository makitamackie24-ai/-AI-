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
st.write("過去3年分のデータから、グランビルの法則、ダウ理論、一目均衡表などの主要テクニカル指標と、AI（ランダムフォレスト200本）を用いた1ヶ月後のトレンド予測を統合して表示します。")

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

    # GC / DC
    df['Sig_GC'] = (df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))
    df['Sig_DC'] = (df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))
    
    # MACD
    df['Sig_MACD_Buy'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    df['Sig_MACD_Sell'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    # グランビルの法則 (5日線と20日線による厳密な判定)
    sma20_diff = df['SMA_20'] - df['SMA_20'].shift(1)
    sma20_diff_past = df['SMA_20'].shift(1) - df['SMA_20'].shift(6) # 過去5日間の傾き
    sma5_diff = df['SMA_5'] - df['SMA_5'].shift(1)
    sma5_diff_prev = df['SMA_5'].shift(1) - df['SMA_5'].shift(2)
    
    cross_up = (df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))
    cross_dn = (df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))
    
    # 買い①（買い転換）：20日線が一定期間下落後、横ばいor上向きで、5日線が下から上へ抜ける
    g_buy1 = cross_up & (sma20_diff_past < 0) & (sma20_diff >= 0)
    # 買い②（押し目買い）：20日線が上向きの時、5日線が下から上へ抜ける
    g_buy2 = cross_up & (sma20_diff > 0)
    # 買い③（買い乗せ）：20日線が上向きの時、5日線が下落するも20日線を下抜けず再度上昇
    g_buy3 = (sma20_diff > 0) & (df['SMA_5'] > df['SMA_20']) & (sma5_diff_prev < 0) & (sma5_diff > 0)
    
    df['Sig_Granville_Buy_1'] = g_buy1
    df['Sig_Granville_Buy_2'] = g_buy2
    df['Sig_Granville_Buy_3'] = g_buy3
    df['Sig_Granville_Buy'] = g_buy1 | g_buy2 | g_buy3
    
    # 売り①（売り転換）：20日線が一定期間上昇後、横ばいor下向きで、5日線が上から下へ抜ける
    g_sell1 = cross_dn & (sma20_diff_past > 0) & (sma20_diff <= 0)
    # 売り②（戻り売り）：20日線が下向きの時、5日線が上から下へ抜ける
    g_sell2 = cross_dn & (sma20_diff < 0)
    # 売り③（売り乗せ）：20日線が下向きの時、5日線が上昇するも20日線を上抜けず再度下落
    g_sell3 = (sma20_diff < 0) & (df['SMA_5'] < df['SMA_20']) & (sma5_diff_prev > 0) & (sma5_diff < 0)
    
    df['Sig_Granville_Sell_1'] = g_sell1
    df['Sig_Granville_Sell_2'] = g_sell2
    df['Sig_Granville_Sell_3'] = g_sell3
    df['Sig_Granville_Sell'] = g_sell1 | g_sell2 | g_sell3
    
    # 一目均衡表 (厳密な三役好転・三役逆転の判定)
    cloud_top = df[['Senkou1', 'Senkou2']].max(axis=1)
    cloud_bottom = df[['Senkou1', 'Senkou2']].min(axis=1)
    
    # --- 買い（三役好転）状態の定義 ---
    # 1. 転換線が基準線を上回る
    cond_tenkan_up = df['Tenkan'] > df['Kijun']
    # 2. 遅行スパン(当日の終値)が、26日前のローソク足の上限(高値)を上回る
    cond_chikou_up = df['Close'] > df['High'].shift(26)
    # 3. ローソク足が雲の上限を上回る
    cond_kumo_up = df['Close'] > cloud_top
    
    all_up_today = cond_tenkan_up & cond_chikou_up & cond_kumo_up
    
    # いずれかの条件が「今日」新たに成立（クロス）したか
    cross_tenkan_up = cond_tenkan_up & (df['Tenkan'].shift(1) <= df['Kijun'].shift(1))
    cross_chikou_up = cond_chikou_up & (df['Close'].shift(1) <= df['High'].shift(27)) # 昨日の終値が、実質27日前の高値以下
    cross_kumo_up = cond_kumo_up & (df['Close'].shift(1) <= cloud_top.shift(1))
    
    # 三役好転: 3つの条件を全て満たし、かつ、今日どれかの条件が新たにクロスして達成された日のみ検出
    df['Sig_Ichimoku_Buy'] = all_up_today & (cross_tenkan_up | cross_chikou_up | cross_kumo_up)
    
    # --- 売り（三役逆転）状態の定義 ---
    # 1. 転換線が基準線を下回る
    cond_tenkan_dn = df['Tenkan'] < df['Kijun']
    # 2. 遅行スパン(当日の終値)が、26日前のローソク足の下限(安値)を下回る
    cond_chikou_dn = df['Close'] < df['Low'].shift(26)
    # 3. ローソク足が雲の下限を下回る
    cond_kumo_dn = df['Close'] < cloud_bottom
    
    all_dn_today = cond_tenkan_dn & cond_chikou_dn & cond_kumo_dn
    
    cross_tenkan_dn = cond_tenkan_dn & (df['Tenkan'].shift(1) >= df['Kijun'].shift(1))
    cross_chikou_dn = cond_chikou_dn & (df['Close'].shift(1) >= df['Low'].shift(27))
    cross_kumo_dn = cond_kumo_dn & (df['Close'].shift(1) >= cloud_bottom.shift(1))
    
    # 三役逆転: 3つの条件を全て満たし、かつ、今日どれかの条件が新たにクロスして達成された日のみ検出
    df['Sig_Ichimoku_Sell'] = all_dn_today & (cross_tenkan_dn | cross_chikou_dn | cross_kumo_dn)
    
    # ダウ理論 (簡易ベクトル判定)
    df['Local_Max'] = df['High'][(df['High'] == df['High'].rolling(5, center=True).max())]
    df['Local_Min'] = df['Low'][(df['Low'] == df['Low'].rolling(5, center=True).min())]
    df['Last_Max'] = df['Local_Max'].ffill()
    df['Last_Min'] = df['Local_Min'].ffill()
    
    max_s = df['Local_Max'].dropna()
    min_s = df['Local_Min'].dropna()
    df['Prev_Max'] = np.nan
    df['Prev_Min'] = np.nan
    if len(max_s) >= 2: df.loc[max_s.index, 'Prev_Max'] = max_s.shift(1)
    if len(min_s) >= 2: df.loc[min_s.index, 'Prev_Min'] = min_s.shift(1)
    df['Prev_Max'] = df['Prev_Max'].ffill()
    df['Prev_Min'] = df['Prev_Min'].ffill()
    
    dow_up_cond = (df['Last_Max'] > df['Prev_Max']) & (df['Last_Min'] > df['Prev_Min'])
    dow_dn_cond = (df['Last_Max'] < df['Prev_Max']) & (df['Last_Min'] < df['Prev_Min'])
    df['Sig_Dow_Buy'] = dow_up_cond & (df['Local_Max'].notna() | df['Local_Min'].notna()) & ~dow_up_cond.shift(1).fillna(False)
    df['Sig_Dow_Sell'] = dow_dn_cond & (df['Local_Max'].notna() | df['Local_Min'].notna()) & ~dow_dn_cond.shift(1).fillna(False)

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
    if len(df) < 52 or 'Sig_Ichimoku_Buy' not in df.columns:
        return False, False, None
        
    # 直近5日間で三役好転/逆転の「シグナル」が点灯し、
    # かつ最新日現在もその「状態」が崩れずに継続しているかを判定する
    recent_buy_sigs = df['Sig_Ichimoku_Buy'].tail(5)
    recent_sell_sigs = df['Sig_Ichimoku_Sell'].tail(5)
    
    # 最新日の「状態」を確認
    cloud_top = max(df['Senkou1'].iloc[-1], df['Senkou2'].iloc[-1])
    cloud_bottom = min(df['Senkou1'].iloc[-1], df['Senkou2'].iloc[-1])
    
    is_up_state = (df['Tenkan'].iloc[-1] > df['Kijun'].iloc[-1]) and \
                  (df['Close'].iloc[-1] > df['High'].iloc[-26]) and \
                  (df['Close'].iloc[-1] > cloud_top)
                  
    is_dn_state = (df['Tenkan'].iloc[-1] < df['Kijun'].iloc[-1]) and \
                  (df['Close'].iloc[-1] < df['Low'].iloc[-26]) and \
                  (df['Close'].iloc[-1] < cloud_bottom)
    
    up_turn = False
    dn_turn = False
    trigger_date = None
    
    if is_up_state and recent_buy_sigs.any():
        up_turn = True
        # 直近5日間の中で最後にシグナルが点灯した日付を取得
        trigger_idx = recent_buy_sigs[recent_buy_sigs == True].index[-1]
        trigger_date = pd.to_datetime(trigger_idx).strftime('%Y-%m-%d')
        
    elif is_dn_state and recent_sell_sigs.any():
        dn_turn = True
        trigger_idx = recent_sell_sigs[recent_sell_sigs == True].index[-1]
        trigger_date = pd.to_datetime(trigger_idx).strftime('%Y-%m-%d')
    
    return up_turn, dn_turn, trigger_date

# --- AI予測モデル学習・推論 ---
def run_ai_prediction(df):
    # 未来（20営業日後）のトレンドを判定するための特徴量をシフトして作成
    df['Future_Close'] = df['Close'].shift(-20)
    df['Future_SMA_20'] = df['SMA_20'].shift(-20)
    df['Future_Past5_SMA_20'] = df['SMA_20'].shift(-15) # 20日後の時点での「5日前の20日線」
    
    # 未来の20日線の傾き
    df['Future_SMA20_Slope'] = (df['Future_SMA_20'] - df['Future_Past5_SMA_20']) / df['Future_Past5_SMA_20'] * 100
    
    # 未来のトレンドラベル付け (1: 上昇, -1: 下落, 0: ボックス)
    conditions = [
        (df['Future_SMA20_Slope'] > 0.3) & (df['Future_Close'] > df['Future_SMA_20']),
        (df['Future_SMA20_Slope'] < -0.3) & (df['Future_Close'] < df['Future_SMA_20'])
    ]
    choices = [1, -1]
    df['Future_Trend'] = np.select(conditions, choices, default=0)
    
    # 欠損値を含む行（直近20日分などは未来がNaNになるため）の除外用のマスク
    mask = df['Future_Close'].notna() & df['Future_SMA_20'].notna() & df['Future_Past5_SMA_20'].notna()
    
    features = ['Close', 'SMA_5', 'SMA_20', 'MACD', 'MACD_Signal', 'RSI']
    
    # NaNを含む行を除外して学習データ作成
    train_df = df[mask].dropna(subset=features).copy()
    
    if len(train_df) < 100:
        return 0.0, 0.0, 0.0
        
    X = train_df[features]
    y = train_df['Future_Trend']
    
    # 3クラス分類モデル (木200本)
    classes = y.unique()
    if len(classes) > 1:
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model.fit(X, y)
        
        # 予測
        probas = model.predict_proba(df[features].iloc[-1:])
        
        # predict_probaの結果は model.classes_ の順序に従うため、辞書化して取得
        prob_dict = {cls: prob for cls, prob in zip(model.classes_, probas[0])}
        
        prob_up = prob_dict.get(1, 0.0) * 100
        prob_box = prob_dict.get(0, 0.0) * 100
        prob_down = prob_dict.get(-1, 0.0) * 100
    else:
        # 学習データが全て同じクラスの場合
        prob_up = 100.0 if classes[0] == 1 else 0.0
        prob_box = 100.0 if classes[0] == 0 else 0.0
        prob_down = 100.0 if classes[0] == -1 else 0.0
        
    return prob_up, prob_box, prob_down

# --- シグナルスコア計算関数 ---
def calculate_signal_score(signal_name, stats, is_buy):
    """
    バックテストの平均変動率から、各シグナルの性質（中長期・短中期・短期）に
    合わせたウェイト付けを行い、0〜100点のスコアを算出する。
    """
    if not stats or pd.isna(stats.get('1週間後')):
        return None # データ不足

    score = 0
    
    if is_buy:
        # 買いシグナルの性質定義
        properties = {
            "グランビルの法則(買い①: 買い転換)": "中長期",
            "グランビルの法則(買い②: 押し目買い)": "中長期",
            "グランビルの法則(買い③: 買い乗せ)": "短中期",
            "ゴールデンクロス(5日/20日)": "中長期",
            "MACD上抜け": "中長期",
            "ダウ理論(上昇波)": "短期",
            "一目均衡表(三役好転)": "中長期"
        }
        prop = properties.get(signal_name, "中長期")
        
        v1 = stats.get('1週間後', 0)
        v2 = stats.get('2週間後', 0)
        v3 = stats.get('3週間後', 0)
        v4 = stats.get('4週間後', 0)
        
        # 欠損値対応
        v1 = 0 if pd.isna(v1) else v1
        v2 = 0 if pd.isna(v2) else v2
        v3 = 0 if pd.isna(v3) else v3
        v4 = 0 if pd.isna(v4) else v4

        if prop == "短期":
            # 1週間後を最重視、次いで2週間後
            weighted_return = (v1 * 0.7) + (v2 * 0.3)
        elif prop == "短中期":
            # 2週間後、3週間後をバランスよく
            weighted_return = (v1 * 0.1) + (v2 * 0.45) + (v3 * 0.45)
        else: # 中長期
            # 3週間後、4週間後を最重視
            weighted_return = (v2 * 0.1) + (v3 * 0.4) + (v4 * 0.5)
            
        # 変動率(%)を点数化 (例: 平均+3%で約70点、+5%以上で90点超えを想定したロジック)
        # sigmoid的なカーブで0-100に収める簡易計算
        base = max(0, weighted_return) # マイナスリターンは0点ベース
        score = min(100, int(base * 15 + 40)) if base > 0 else 0
        if score > 0 and base > 0.5:
             score = min(100, int(50 + base * 10))

    else:
        # 売りシグナルの性質定義 (買いの反対)
        properties = {
            "グランビルの法則(売り①: 売り転換)": "中長期",
            "グランビルの法則(売り②: 戻り売り)": "中長期",
            "グランビルの法則(売り③: 売り乗せ)": "短中期",
            "デッドクロス(5日/20日)": "中長期",
            "MACD下抜け": "中長期",
            "ダウ理論(下落波)": "短期",
            "一目均衡表(三役逆転)": "中長期"
        }
        prop = properties.get(signal_name, "中長期")
        
        v1 = stats.get('1日後', 0)
        v2 = stats.get('2日後', 0)
        v3 = stats.get('3日後', 0)
        vw = stats.get('1週間後', 0)
        
        # 欠損値対応
        v1 = 0 if pd.isna(v1) else v1
        v2 = 0 if pd.isna(v2) else v2
        v3 = 0 if pd.isna(v3) else v3
        vw = 0 if pd.isna(vw) else vw
        
        # 売りは下落（マイナス）で成功なので、符号を反転させて評価
        v1, v2, v3, vw = -v1, -v2, -v3, -vw

        if prop == "短期":
            # 1〜3日後を重視
            weighted_return = (v1 * 0.4) + (v2 * 0.4) + (v3 * 0.2)
        elif prop == "短中期":
            # 3日後、1週間後を重視
            weighted_return = (v3 * 0.4) + (vw * 0.6)
        else: # 中長期
            # 1週間後を最重視
            weighted_return = (v3 * 0.2) + (vw * 0.8)
            
        base = max(0, weighted_return)
        score = min(100, int(base * 15 + 40)) if base > 0 else 0
        if score > 0 and base > 0.5:
             score = min(100, int(50 + base * 10))
             
    # 少なすぎるサンプル（点灯回数1回などで極端な値になること）へのペナルティ
    if stats.get('点灯回数', 0) <= 2:
        score = int(score * 0.7)

    return score


# --- メイン解析処理 ---
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_all_stocks():
    results = []
    
    # 過去3年分のデータ取得 (Step0)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 3)
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
            # グランビルの法則 (直近1ヶ月間（20営業日）でシグナルが点灯しているか判定)
            recent_g_buy1 = df['Sig_Granville_Buy_1'].tail(20)
            recent_g_buy2 = df['Sig_Granville_Buy_2'].tail(20)
            recent_g_buy3 = df['Sig_Granville_Buy_3'].tail(20)
            
            recent_g_sell1 = df['Sig_Granville_Sell_1'].tail(20)
            recent_g_sell2 = df['Sig_Granville_Sell_2'].tail(20)
            recent_g_sell3 = df['Sig_Granville_Sell_3'].tail(20)
            
            def get_signal_info(series):
                active = bool(series.any())
                date = pd.to_datetime(series[series == True].index[-1]).strftime('%Y-%m-%d') if active else None
                return active, date
                
            gb1_act, gb1_date = get_signal_info(recent_g_buy1)
            gb2_act, gb2_date = get_signal_info(recent_g_buy2)
            gb3_act, gb3_date = get_signal_info(recent_g_buy3)
            
            gs1_act, gs1_date = get_signal_info(recent_g_sell1)
            gs2_act, gs2_date = get_signal_info(recent_g_sell2)
            gs3_act, gs3_date = get_signal_info(recent_g_sell3)
                    
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
            
            # AI予測
            prob_up, prob_box, prob_down = run_ai_prediction(df)
            
            # --- 過去半年間のシグナル実績（平均変動率）の計算 ---
            # 買いリターン計算 (5, 10, 15, 20営業日後)
            df['Ret_Buy_5'] = (df['Close'].shift(-5) - df['Close']) / df['Close'] * 100
            df['Ret_Buy_10'] = (df['Close'].shift(-10) - df['Close']) / df['Close'] * 100
            df['Ret_Buy_15'] = (df['Close'].shift(-15) - df['Close']) / df['Close'] * 100
            df['Ret_Buy_20'] = (df['Close'].shift(-20) - df['Close']) / df['Close'] * 100
            
            # 売りリターン計算 (1, 2, 3, 5営業日後)
            df['Ret_Sell_1'] = (df['Close'].shift(-1) - df['Close']) / df['Close'] * 100
            df['Ret_Sell_2'] = (df['Close'].shift(-2) - df['Close']) / df['Close'] * 100
            df['Ret_Sell_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close'] * 100
            df['Ret_Sell_5'] = (df['Close'].shift(-5) - df['Close']) / df['Close'] * 100
            
            df_half = df.tail(150) # 過去半年分（約150営業日）を抽出
            
            buy_cols = {
                "グランビルの法則(買い①: 買い転換)": "Sig_Granville_Buy_1",
                "グランビルの法則(買い②: 押し目買い)": "Sig_Granville_Buy_2",
                "グランビルの法則(買い③: 買い乗せ)": "Sig_Granville_Buy_3",
                "ゴールデンクロス(5日/20日)": "Sig_GC",
                "MACD上抜け": "Sig_MACD_Buy",
                "ダウ理論(上昇波)": "Sig_Dow_Buy",
                "一目均衡表(三役好転)": "Sig_Ichimoku_Buy"
            }
            
            buy_stats = []
            buy_scores = {}
            for s_name, col_name in buy_cols.items():
                if col_name in df_half.columns:
                    rows = df_half[df_half[col_name] == True]
                    cnt = len(rows)
                    if cnt > 0:
                        stat_dict = {
                            "シグナル": s_name,
                            "点灯回数": cnt,
                            "1週間後": rows['Ret_Buy_5'].mean(),
                            "2週間後": rows['Ret_Buy_10'].mean(),
                            "3週間後": rows['Ret_Buy_15'].mean(),
                            "4週間後": rows['Ret_Buy_20'].mean()
                        }
                        # スコア計算
                        score = calculate_signal_score(s_name, stat_dict, is_buy=True)
                        stat_dict["精度スコア"] = score
                        buy_scores[s_name] = score
                        
                        buy_stats.append(stat_dict)
                        
            sell_cols = {
                "グランビルの法則(売り①: 売り転換)": "Sig_Granville_Sell_1",
                "グランビルの法則(売り②: 戻り売り)": "Sig_Granville_Sell_2",
                "グランビルの法則(売り③: 売り乗せ)": "Sig_Granville_Sell_3",
                "デッドクロス(5日/20日)": "Sig_DC",
                "MACD下抜け": "Sig_MACD_Sell",
                "ダウ理論(下落波)": "Sig_Dow_Sell",
                "一目均衡表(三役逆転)": "Sig_Ichimoku_Sell"
            }
            
            sell_stats = []
            sell_scores = {}
            for s_name, col_name in sell_cols.items():
                if col_name in df_half.columns:
                    rows = df_half[df_half[col_name] == True]
                    cnt = len(rows)
                    if cnt > 0:
                        stat_dict = {
                            "シグナル": s_name,
                            "点灯回数": cnt,
                            "1日後": rows['Ret_Sell_1'].mean(),
                            "2日後": rows['Ret_Sell_2'].mean(),
                            "3日後": rows['Ret_Sell_3'].mean(),
                            "1週間後": rows['Ret_Sell_5'].mean()
                        }
                        # スコア計算
                        score = calculate_signal_score(s_name, stat_dict, is_buy=False)
                        stat_dict["精度スコア"] = score
                        sell_scores[s_name] = score
                        
                        sell_stats.append(stat_dict)
            
            # 精度スコア列を整える（表示順序用）
            if buy_stats:
                # 精度スコア列を2番目に移動
                for i in range(len(buy_stats)):
                     score_val = buy_stats[i].pop("精度スコア")
                     new_dict = {"シグナル": buy_stats[i]["シグナル"], "精度スコア": score_val}
                     new_dict.update(buy_stats[i])
                     buy_stats[i] = new_dict

            if sell_stats:
                for i in range(len(sell_stats)):
                     score_val = sell_stats[i].pop("精度スコア")
                     new_dict = {"シグナル": sell_stats[i]["シグナル"], "精度スコア": score_val}
                     new_dict.update(sell_stats[i])
                     sell_stats[i] = new_dict

            
            # --- 直近1週間（過去5営業日）の株価データを取得 ---
            recent_ohlc = df.tail(5)[['Open', 'High', 'Low', 'Close']].copy()
            recent_ohlc.index = recent_ohlc.index.strftime('%Y-%m-%d')
            recent_ohlc.columns = ['始値', '高値', '安値', '終値']
            recent_ohlc.index.name = '日付'
            
            results.append({
                "name": name,
                "ticker": ticker,
                "price": curr_close,
                "recent_ohlc": recent_ohlc,
                "trend": current_trend,
                "is_box": "ボックストレンド" in current_trend,
                "prob_up": prob_up,
                "prob_box": prob_box,
                "prob_down": prob_down,
                "signals_buy": {
                    "グランビルの法則(買い①: 買い転換)": {"active": gb1_act, "date": gb1_date},
                    "グランビルの法則(買い②: 押し目買い)": {"active": gb2_act, "date": gb2_date},
                    "グランビルの法則(買い③: 買い乗せ)": {"active": gb3_act, "date": gb3_date},
                    "ゴールデンクロス(5日/20日)": {"active": gc, "date": latest_date_str},
                    "MACD上抜け": {"active": macd_buy, "date": latest_date_str},
                    "ダウ理論(上昇波)": {"active": dow_up, "date": dow_date},
                    "一目均衡表(三役好転)": {"active": ichimoku_up, "date": ichimoku_date}
                },
                "signals_sell": {
                    "グランビルの法則(売り①: 売り転換)": {"active": gs1_act, "date": gs1_date},
                    "グランビルの法則(売り②: 戻り売り)": {"active": gs2_act, "date": gs2_date},
                    "グランビルの法則(売り③: 売り乗せ)": {"active": gs3_act, "date": gs3_date},
                    "デッドクロス(5日/20日)": {"active": dc, "date": latest_date_str},
                    "MACD下抜け": {"active": macd_sell, "date": latest_date_str},
                    "ダウ理論(下落波)": {"active": dow_down, "date": dow_date},
                    "一目均衡表(三役逆転)": {"active": ichimoku_down, "date": ichimoku_date}
                },
                "buy_stats": buy_stats,
                "buy_scores": buy_scores,
                "sell_stats": sell_stats,
                "sell_scores": sell_scores,
                "chart_data": df.tail(150).copy() # チャート描画用に直近約半年分のデータを保存
            })
            
        except Exception as e:
            st.error(f"{name} のデータ処理中にエラーが発生しました: {e}")
            
        progress_bar.progress((i + 1) / len(tickers))
        
    progress_bar.empty()
    return results

def format_score(val):
    if pd.isna(val):
         return "-"
    color = "green" if val >= 70 else "black" if val >= 40 else "red"
    return f'<span style="color: {color}; font-weight: bold;">{int(val)}点</span>'


# --- 画面描画 ---
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("全10銘柄の総合診断を実行 (過去3年分データ取得・AI学習)", type="primary"):
        with st.spinner("過去3年分のデータを取得し、AI（ランダムフォレスト200本）と各種テクニカル指標の解析を行っています..."):
            results = analyze_all_stocks()
            st.session_state['results'] = results
with col2:
    if st.button("🔄 データを最新に更新 (キャッシュクリア)"):
        st.cache_data.clear()
        if 'results' in st.session_state:
            del st.session_state['results']
        st.rerun()

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
                st.markdown("##### 🤖 AI予測 (1ヶ月後のトレンド確率)")
                st.markdown(f"<span style='color: green; font-weight: bold;'>上昇トレンドになる確率: {res['prob_up']:.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: gray; font-weight: bold;'>ボックストレンドになる確率: {res['prob_box']:.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: red; font-weight: bold;'>下落トレンドになる確率: {res['prob_down']:.1f}%</span>", unsafe_allow_html=True)

            st.markdown("##### 📅 直近1週間の株価推移")
            # 5営業日分のデータをコンパクトなテーブルで表示
            st.dataframe(
                res['recent_ohlc'].style.format("¥{:,.0f}"), 
                use_container_width=True, 
                height=212
            )

            st.markdown("---")
            
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                st.markdown("#### 🔵 Step2: 買いタイミングのシグナル")
                buy_count = 0
                for sig_name, sig_data in res['signals_buy'].items():
                    if sig_data['active']:
                        date_str = f" [{sig_data['date']}]" if sig_data['date'] else ""
                        
                        # スコアの取得と表示
                        score = res['buy_scores'].get(sig_name)
                        score_str = f" <span style='font-size: 0.8em;'>(精度: {int(score)}点)</span>" if score is not None else ""
                        
                        st.markdown(f"- ✅ **{sig_name}** 点灯 {date_str}{score_str}", unsafe_allow_html=True)
                        buy_count += 1
                if buy_count == 0:
                    st.write("現在、点灯している買いシグナルはありません。")
                    
            with col_sell:
                st.markdown("#### 🔴 Step3: 売りタイミングのシグナル")
                sell_count = 0
                for sig_name, sig_data in res['signals_sell'].items():
                    if sig_data['active']:
                        date_str = f" [{sig_data['date']}]" if sig_data['date'] else ""
                        
                        score = res['sell_scores'].get(sig_name)
                        score_str = f" <span style='font-size: 0.8em;'>(精度: {int(score)}点)</span>" if score is not None else ""
                        
                        st.markdown(f"- ⚠️ **{sig_name}** 点灯 {date_str}{score_str}", unsafe_allow_html=True)
                        sell_count += 1
                if sell_count == 0:
                    st.write("現在、点灯している売りシグナルはありません。")

            # --- バックテスト結果の表示セクション ---
            with st.expander("📈 過去半年間のシグナル実績（平均変動割合）と精度スコア"):
                st.markdown("""
                過去半年間に各シグナルが点灯した際、その後の終値が平均で何％変動したかを表示します。（＋なら上昇、－なら下落）  
                **【精度スコアについて(100点満点)】**  
                シグナルの性質（ダウ理論なら短期、MACDなら中長期など）に合わせて重視する期間の変動率を加重平均し、その銘柄においてどの程度「期待通りに機能しているか」を独自に点数化したものです。（70点以上は高精度）
                """)
                
                st.markdown("##### 🔵 買いシグナルの実績 (1〜4週間後)")
                if res.get('buy_stats'):
                    df_buy_stats = pd.DataFrame(res['buy_stats'])
                    st.dataframe(
                        df_buy_stats.style.format({
                            "精度スコア": "{:.0f}",
                            "1週間後": "{:+.2f}%", 
                            "2週間後": "{:+.2f}%", 
                            "3週間後": "{:+.2f}%", 
                            "4週間後": "{:+.2f}%"
                        }, na_rep="-").background_gradient(subset=['精度スコア'], cmap='Greens', vmin=0, vmax=100),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("過去半年に点灯した買いシグナルはありません。")
                    
                st.markdown("##### 🔴 売りシグナルの実績 (1〜3日後・1週間後)")
                if res.get('sell_stats'):
                    df_sell_stats = pd.DataFrame(res['sell_stats'])
                    st.dataframe(
                        df_sell_stats.style.format({
                            "精度スコア": "{:.0f}",
                            "1日後": "{:+.2f}%", 
                            "2日後": "{:+.2f}%", 
                            "3日後": "{:+.2f}%",
                            "1週間後": "{:+.2f}%"
                        }, na_rep="-").background_gradient(subset=['精度スコア'], cmap='Reds', vmin=0, vmax=100),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("過去半年に点灯した売りシグナルはありません。")

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
                
                # チャート用シグナル抽出
                buy_x, buy_y, buy_text = [], [], []
                sell_x, sell_y, sell_text = [], [], []
                dow_buy_x, dow_buy_y, dow_buy_text = [], [], []
                dow_sell_x, dow_sell_y, dow_sell_text = [], [], []

                for idx, row in df_c.iterrows():
                    d = idx.strftime('%Y-%m-%d')
                    
                    # 買いシグナル
                    b_sigs = []
                    if pd.notna(row.get('Sig_GC')) and bool(row.get('Sig_GC')): b_sigs.append("ゴールデンクロス(5日/20日)")
                    if pd.notna(row.get('Sig_MACD_Buy')) and bool(row.get('Sig_MACD_Buy')): b_sigs.append("MACD上抜け")
                    if pd.notna(row.get('Sig_Granville_Buy_1')) and bool(row.get('Sig_Granville_Buy_1')): b_sigs.append("グランビルの法則(買い①: 買い転換)")
                    if pd.notna(row.get('Sig_Granville_Buy_2')) and bool(row.get('Sig_Granville_Buy_2')): b_sigs.append("グランビルの法則(買い②: 押し目買い)")
                    if pd.notna(row.get('Sig_Granville_Buy_3')) and bool(row.get('Sig_Granville_Buy_3')): b_sigs.append("グランビルの法則(買い③: 買い乗せ)")
                    if pd.notna(row.get('Sig_Ichimoku_Buy')) and bool(row.get('Sig_Ichimoku_Buy')): b_sigs.append("一目均衡表(三役好転)")
                    
                    if b_sigs:
                        buy_x.append(idx)
                        buy_y.append(row['Low'] * 0.96)
                        buy_text.append(f"<b>【買いシグナル】 {d}</b><br>" + "<br>".join(b_sigs))
                        
                    # 売りシグナル
                    s_sigs = []
                    if pd.notna(row.get('Sig_DC')) and bool(row.get('Sig_DC')): s_sigs.append("デッドクロス(5日/20日)")
                    if pd.notna(row.get('Sig_MACD_Sell')) and bool(row.get('Sig_MACD_Sell')): s_sigs.append("MACD下抜け")
                    if pd.notna(row.get('Sig_Granville_Sell_1')) and bool(row.get('Sig_Granville_Sell_1')): s_sigs.append("グランビルの法則(売り①: 売り転換)")
                    if pd.notna(row.get('Sig_Granville_Sell_2')) and bool(row.get('Sig_Granville_Sell_2')): s_sigs.append("グランビルの法則(売り②: 戻り売り)")
                    if pd.notna(row.get('Sig_Granville_Sell_3')) and bool(row.get('Sig_Granville_Sell_3')): s_sigs.append("グランビルの法則(売り③: 売り乗せ)")
                    if pd.notna(row.get('Sig_Ichimoku_Sell')) and bool(row.get('Sig_Ichimoku_Sell')): s_sigs.append("一目均衡表(三役逆転)")
                    
                    if s_sigs:
                        sell_x.append(idx)
                        sell_y.append(row['High'] * 1.04)
                        sell_text.append(f"<b>【売りシグナル】 {d}</b><br>" + "<br>".join(s_sigs))
                        
                    # ダウ理論買い
                    if pd.notna(row.get('Sig_Dow_Buy')) and bool(row.get('Sig_Dow_Buy')):
                        dow_buy_x.append(idx)
                        dow_buy_y.append(row['Low'] * 0.90) # 通常シグナルと被らないようさらに下に配置
                        dow_buy_text.append(f"<b>【ダウ理論 買い】 {d}</b><br>上昇トレンド転換(高値・安値切り上げ)")

                    # ダウ理論売り
                    if pd.notna(row.get('Sig_Dow_Sell')) and bool(row.get('Sig_Dow_Sell')):
                        dow_sell_x.append(idx)
                        dow_sell_y.append(row['High'] * 1.10) # 通常シグナルと被らないようさらに上に配置
                        dow_sell_text.append(f"<b>【ダウ理論 売り】 {d}</b><br>下落トレンド転換(高値・安値切り下げ)")
                        
                # 買いシグナル (水色に変更)
                if buy_x:
                    fig.add_trace(go.Scatter(
                        x=buy_x, y=buy_y, mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='#00BCD4', line=dict(width=1, color='white')),
                        name='買いシグナル(各指標)', hovertext=buy_text, hoverinfo='text'
                    ), row=1, col=1)
                        
                # 売りシグナル (ピンク色に変更)
                if sell_x:
                    fig.add_trace(go.Scatter(
                        x=sell_x, y=sell_y, mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='#E91E63', line=dict(width=1, color='white')),
                        name='売りシグナル(各指標)', hovertext=sell_text, hoverinfo='text'
                    ), row=1, col=1)

                if dow_buy_x:
                    fig.add_trace(go.Scatter(
                        x=dow_buy_x, y=dow_buy_y, mode='markers',
                        marker=dict(symbol='star', size=16, color='#FFC107', line=dict(width=1, color='black')),
                        name='ダウ理論 買い転換', hovertext=dow_buy_text, hoverinfo='text'
                    ), row=1, col=1)

                if dow_sell_x:
                    fig.add_trace(go.Scatter(
                        x=dow_sell_x, y=dow_sell_y, mode='markers',
                        marker=dict(symbol='star', size=16, color='#9C27B0', line=dict(width=1, color='white')),
                        name='ダウ理論 売り転換', hovertext=dow_sell_text, hoverinfo='text'
                    ), row=1, col=1)
                
                # レイアウト調整 (凡例を左上に移動)
                fig.update_layout(
                    height=500, 
                    margin=dict(l=0, r=0, t=30, b=0), 
                    xaxis_rangeslider_visible=False, # ローソク足標準のレンジスライダーを非表示
                    showlegend=True,
                    legend=dict(
                        orientation="h", 
                        yanchor="top", 
                        y=1.02, 
                        xanchor="left", 
                        x=0,
                        bgcolor="rgba(255, 255, 255, 0.5)" # 凡例の背景を少し透過
                    )
                )
                
                # 土日などの休場日の空白を詰める処理
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                
                st.plotly_chart(fig, use_container_width=True)
