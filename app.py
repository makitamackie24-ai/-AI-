import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="多角的シグナル検知AI", layout="wide")

def calculate_signals(df):
    """各種テクニカル指標とシグナルを計算する関数"""
    df = df.copy()
    
    # yfinanceのバージョンや取得状況によって列がMultiIndexになる場合の対策
    if isinstance(df.columns, pd.MultiIndex):
        # 必要な列だけを抽出し、フラットなSeriesにする
        open_col = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
        high_col = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low_col = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close_col = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        # 新しいDataFrameとして再構築
        df = pd.DataFrame({
            'Open': open_col,
            'High': high_col,
            'Low': low_col,
            'Close': close_col
        })
    else:
        # 万が一DataFrameになっている列があればSeriesにする
        for col in ['Open', 'High', 'Low', 'Close']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

    # 数値型に明示的に変換（エラー防止）
    df['Close'] = pd.to_numeric(df['Close'])
    df['Open'] = pd.to_numeric(df['Open'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean() # パーフェクトオーダー用
    
    # --- MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- RSI ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 一目均衡表 ---
    df['Tenkan'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou'] = df['Close'].shift(-26)
    
    # --- グランビルの法則 (5日線と20日線の関係) ---
    sma20_diff = df['SMA_20'].diff()
    sma20_is_up = sma20_diff > 0
    sma20_is_down = sma20_diff < 0
    sma20_is_flat = sma20_diff.abs() < (df['SMA_20'] * 0.001) # 0.1%未満の変動を横ばいとする
    
    cross_up = (df['SMA_5'] > df['SMA_20']) & (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))
    cross_down = (df['SMA_5'] < df['SMA_20']) & (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))
    
    # 買い① (転換): 20日線が下落or横ばい後、5日線が下から上へ
    df['Sig_Granville_Buy_1'] = cross_up & (sma20_is_down | sma20_is_flat).shift(1)
    
    # 買い② (押し目買い): 20日線上向き、5日線が20日線を割り、再度上抜け
    # (ここでは簡易的に、20日線上向きでのクロスアップとする)
    df['Sig_Granville_Buy_2'] = cross_up & sma20_is_up.shift(1)
    
    # 買い③ (買い乗せ): 20日線上向き、5日線が下落するも20日線を割らずに再上昇
    sma5_diff = df['SMA_5'].diff()
    df['Sig_Granville_Buy_3'] = sma20_is_up & (df['SMA_5'] > df['SMA_20']) & (sma5_diff > 0) & (sma5_diff.shift(1) < 0)
    
    # 売り① (転換): 20日線が上昇or横ばい後、5日線が上から下へ
    df['Sig_Granville_Sell_1'] = cross_down & (sma20_is_up | sma20_is_flat).shift(1)
    
    # 売り② (戻り売り): 20日線下向き、5日線が20日線を上回り、再度下抜け
    df['Sig_Granville_Sell_2'] = cross_down & sma20_is_down.shift(1)
    
    # 売り③ (売り乗せ): 20日線下向き、5日線が上昇するも20日線を超えずに再下落
    df['Sig_Granville_Sell_3'] = sma20_is_down & (df['SMA_5'] < df['SMA_20']) & (sma5_diff < 0) & (sma5_diff.shift(1) > 0)

    # --- 基本のクロス ---
    df['Sig_GC'] = cross_up # 買い②と被るが汎用シグナルとして保持
    df['Sig_DC'] = cross_down
    
    macd_cross_up = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    macd_cross_down = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    df['Sig_MACD_Buy'] = macd_cross_up
    df['Sig_MACD_Sell'] = macd_cross_down

    # --- パーフェクトオーダー (買い) ---
    po_condition = (df['SMA_5'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_60'])
    # 新規に完成したタイミングのみをシグナルとする
    df['Sig_PO_Buy'] = po_condition & ~po_condition.shift(1).fillna(False)

    # --- 新高値ブレイクアウト (過去半年=125日) ---
    half_year_high = df['High'].shift(1).rolling(window=125).max()
    is_breakout = df['Close'] > half_year_high
    df['Sig_Breakout_Buy'] = is_breakout & ~is_breakout.shift(1).fillna(False)

    # --- ローソク足パターン (売り) ---
    prev_open, prev_close = df['Open'].shift(1), df['Close'].shift(1)
    curr_open, curr_close = df['Open'], df['Close']
    is_prev_bull = prev_close > prev_open # 前日が陽線
    is_curr_bear = curr_close < curr_open # 当日が陰線
    high_zone = df['Close'] > df['SMA_20'] # 高値圏
    
    # 包み足(陰線)
    df['Sig_Engulfing_Sell'] = is_prev_bull & is_curr_bear & (curr_open > prev_close) & (curr_close < prev_open) & high_zone
    # 否定陰線
    df['Sig_Negation_Sell'] = is_prev_bull & is_curr_bear & (curr_open <= prev_close) & (curr_close < prev_open) & high_zone

    # --- 一目均衡表 (厳密な三役好転/逆転) ---
    # 転換線・基準線
    ichimoku_cross_up = df['Tenkan'] > df['Kijun']
    ichimoku_cross_down = df['Tenkan'] < df['Kijun']
    # 雲抜け
    cloud_top = df[['Senkou_A', 'Senkou_B']].max(axis=1)
    cloud_bottom = df[['Senkou_A', 'Senkou_B']].min(axis=1)
    above_cloud = df['Close'] > cloud_top
    below_cloud = df['Close'] < cloud_bottom
    
    # 遅行スパン (26日前のローソク足の高値・安値と比較)
    past_high = df['High'].shift(26)
    past_low = df['Low'].shift(26)
    # 現在の株価(Close)が、26日前の株価と比較してどうか（遅行スパンの現在位置での評価）
    # ※ 本来の遅行スパンは過去にプロットしますが、判定は「現在」行います
    chikou_up = df['Close'] > past_high
    chikou_down = df['Close'] < past_low

    # 3条件が揃っている状態
    sanyaku_koten_state = ichimoku_cross_up & above_cloud & chikou_up
    sanyaku_gyakuten_state = ichimoku_cross_down & below_cloud & chikou_down
    
    # 今日新しく揃った瞬間だけをシグナル化
    df['Sig_Ichimoku_Buy'] = sanyaku_koten_state & ~sanyaku_koten_state.shift(1).fillna(False)
    df['Sig_Ichimoku_Sell'] = sanyaku_gyakuten_state & ~sanyaku_gyakuten_state.shift(1).fillna(False)

    # --- ダウ理論判定 (簡易ジグザグ) ---
    df['Swing_High'] = df['High'][(df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))]
    df['Swing_Low'] = df['Low'][(df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))]
    
    df['Sig_Dow_Buy'] = False
    df['Sig_Dow_Sell'] = False
    
    last_high = last_low = None
    prev_high = prev_low = None
    
    for i in range(1, len(df)):
        if pd.notna(df['Swing_High'].iloc[i]):
            prev_high = last_high
            last_high = df['Swing_High'].iloc[i]
        if pd.notna(df['Swing_Low'].iloc[i]):
            prev_low = last_low
            last_low = df['Swing_Low'].iloc[i]
            
        # 高値切り上げ ＆ 安値切り上げ
        if prev_high and prev_low and last_high and last_low:
            if last_high > prev_high and last_low > prev_low:
                df['Sig_Dow_Buy'].iloc[i] = True
            # 高値切り下げ ＆ 安値切り下げ
            elif last_high < prev_high and last_low < prev_low:
                df['Sig_Dow_Sell'].iloc[i] = True
                
    # 連続を省く
    df['Sig_Dow_Buy'] = df['Sig_Dow_Buy'] & ~df['Sig_Dow_Buy'].shift(1).fillna(False)
    df['Sig_Dow_Sell'] = df['Sig_Dow_Sell'] & ~df['Sig_Dow_Sell'].shift(1).fillna(False)

    return df

def train_predict_trend(df):
    """
    ランダムフォレストを用いて、1ヶ月後（20営業日後）のトレンドを予測
    クラス: 1(上昇), 0(ボックス), -1(下落)
    """
    # 必要な特徴量だけを抽出し、それらにNaNが含まれる行のみを削除する
    # (全体をdropnaすると、一目均衡表の遅行スパン等の影響で直近26日間のデータが全て消えてしまうため)
    features = ['SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'RSI', 'Close']
    df_ml = df[features].copy()
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 100:
        return None, None
        
    X = df_ml[features].copy()
    
    # 目的変数（20日後のSMA20とCloseの関係でトレンドを定義）
    future_sma20 = df_ml['SMA_20'].shift(-20)
    future_close = df_ml['Close'].shift(-20)
    
    # 現在のSMA20に対する20日後のSMA20の変化率
    sma20_change = (future_sma20 - df_ml['SMA_20']) / df_ml['SMA_20']
    
    conditions = [
        (sma20_change > 0.02) & (future_close > future_sma20), # 明確な上昇
        (sma20_change < -0.02) & (future_close < future_sma20) # 明確な下落
    ]
    choices = [1, -1]
    # 条件に合わない場合はボックス(0)
    y = np.select(conditions, choices, default=0)
    
    # 直近20日は正解データがないため学習から除外
    X_train_full = X.iloc[:-20]
    y_train_full = y[:-20]
    
    # 予測対象（最新のデータ）
    X_latest = X.iloc[-1:]
    
    try:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_full, y_train_full)
        
        # 予測確率を取得 [下落(-1), ボックス(0), 上昇(1)] の確率
        # ※クラスが存在しない場合の対策
        classes = model.classes_
        proba = model.predict_proba(X_latest)[0]
        
        prob_dict = {-1: 0.0, 0: 0.0, 1: 0.0}
        for cls, p in zip(classes, proba):
            prob_dict[cls] = p
            
        return prob_dict, model
    except Exception as e:
        return None, None

def calculate_signal_score(df, sig_name, is_buy):
    """
    バックテスト結果に基づき、直近を重視したシグナルの精度スコア(100点満点)を計算
    """
    # 該当シグナルがTrueのインデックスを取得
    sig_idx = df.index[df[sig_name] == True]
    if len(sig_idx) == 0:
        return 0.0
        
    scores = []
    weights = []
    
    # 最新日付からの経過日数でウェイトを減衰させる（直近を重視）
    latest_date = df.index[-1]
    
    # シグナルの性質による評価期間のウェイト
    if is_buy:
        properties = {
            "Sig_Granville_Buy_1": "中長期", "Sig_Granville_Buy_2": "中長期", "Sig_Granville_Buy_3": "短中期",
            "Sig_GC": "中長期", "Sig_PO_Buy": "中長期", "Sig_Breakout_Buy": "中長期",
            "Sig_MACD_Buy": "中長期", "Sig_Dow_Buy": "短期", "Sig_Ichimoku_Buy": "中長期"
        }
    else:
        properties = {
            "Sig_Granville_Sell_1": "中長期", "Sig_Granville_Sell_2": "中長期", "Sig_Granville_Sell_3": "短中期",
            "Sig_DC": "中長期", "Sig_MACD_Sell": "中長期", "Sig_Dow_Sell": "短期", "Sig_Ichimoku_Sell": "中長期",
            "Sig_Engulfing_Sell": "短期", "Sig_Negation_Sell": "短期"
        }
        
    prop = properties.get(sig_name, "中長期")
    
    for idx in sig_idx:
        try:
            current_pos = df.index.get_loc(idx)
            current_price = df['Close'].iloc[current_pos]
            
            # 時間減衰ウェイト (3年前=約1000営業日 でほぼ0に近づく)
            days_diff = (latest_date - idx).days
            time_weight = np.exp(-days_diff / 500) 
            
            if is_buy:
                # 1w, 2w, 3w, 4w (5, 10, 15, 20営業日)
                ret_1w = (df['Close'].iloc[current_pos+5] - current_price)/current_price if current_pos+5 < len(df) else 0
                ret_2w = (df['Close'].iloc[current_pos+10] - current_price)/current_price if current_pos+10 < len(df) else 0
                ret_3w = (df['Close'].iloc[current_pos+15] - current_price)/current_price if current_pos+15 < len(df) else 0
                ret_4w = (df['Close'].iloc[current_pos+20] - current_price)/current_price if current_pos+20 < len(df) else 0
                
                if prop == "短期": val = ret_1w * 0.6 + ret_2w * 0.4
                elif prop == "短中期": val = ret_2w * 0.5 + ret_3w * 0.5
                else: val = ret_3w * 0.4 + ret_4w * 0.6
                
            else:
                # 1d, 2d, 3d, 1w (1, 2, 3, 5営業日)
                # 売りの場合は下がっていればプラス評価
                ret_1d = -(df['Close'].iloc[current_pos+1] - current_price)/current_price if current_pos+1 < len(df) else 0
                ret_2d = -(df['Close'].iloc[current_pos+2] - current_price)/current_price if current_pos+2 < len(df) else 0
                ret_3d = -(df['Close'].iloc[current_pos+3] - current_price)/current_price if current_pos+3 < len(df) else 0
                ret_1w = -(df['Close'].iloc[current_pos+5] - current_price)/current_price if current_pos+5 < len(df) else 0
                
                if prop == "短期": val = ret_1d * 0.3 + ret_2d * 0.3 + ret_3d * 0.4
                elif prop == "短中期": val = ret_3d * 0.5 + ret_1w * 0.5
                else: val = ret_1w * 1.0

            scores.append(val)
            weights.append(time_weight)
        except Exception:
            continue
            
    if not scores or sum(weights) == 0:
        return 0.0
        
    # 加重平均
    weighted_avg = np.average(scores, weights=weights)
    
    # スコア化 (0〜100点)
    # 平均+5%の変動を100点とする
    raw_score = (weighted_avg / 0.05) * 100
    final_score = max(0, min(100, raw_score))
    return final_score

def main():
    st.title("📊 多角的シグナル検知AIツール (3年分析版)")
    st.markdown("直近のシグナル点灯状況、3年間の統計的実績、およびAIによるトレンド予測を統合した分析ダッシュボードです。")
    
    tickers = {
        '日経平均 (日本)': '^N225',
        'TOPIX (日本)': '^TOPX',
        'マザーズ指数 (日本)': '^MOTHERS',
        'S&P 500 (米国)': '^GSPC',
        'NYダウ (米国)': '^DJI',
        'NASDAQ (米国)': '^IXIC',
        'トヨタ自動車': '7203.T',
        '三菱UFJ': '8306.T',
        '三井住友FG': '8316.T',
        'ソフトバンクG': '9984.T',
        'ソニーG': '6758.T',
        'ファーストリテイリング': '9983.T',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'NVIDIA': 'NVDA',
        'Tesla': 'TSLA',
        'Bitcoin (BTC/USD)': 'BTC-USD',
        'USD/JPY (ドル円)': 'JPY=X'
    }
    
    st.sidebar.header("設定")
    selected_name = st.sidebar.selectbox("分析する銘柄を選択", list(tickers.keys()))
    selected_ticker = tickers[selected_name]
    
    if st.sidebar.button("🔄 データを最新に更新 (キャッシュクリア)", type="primary"):
        st.cache_data.clear()
        st.rerun()

    if st.button("🚀 総合診断を実行", type="primary", use_container_width=True):
        with st.spinner(f"{selected_name} の過去3年分のデータを取得・解析中..."):
            
            # --- データ取得 (3年間) ---
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            
            try:
                df = yf.download(selected_ticker, start=start_date, end=end_date)
                if df.empty:
                    st.error("データの取得に失敗しました。")
                    return
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                return
            
            # --- シグナル計算 ---
            df = calculate_signals(df)
            
            # --- 直近のデータとAI予測 ---
            latest_data = df.iloc[-1]
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            prev_data = df.iloc[-2]
            
            prob_dict, model = train_predict_trend(df)
            
            st.markdown("---")
            st.header(f"📈 {selected_name} 総合診断結果 ({latest_date})")
            
            # Step 1: 現在のトレンドとAI予測
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Step 1: 現在のトレンド状態")
                current_price = latest_data['Close']
                sma20 = latest_data['SMA_20']
                sma20_diff = sma20 - prev_data['SMA_20']
                
                if sma20_diff > 0 and current_price > sma20:
                    trend_msg = "🟢 上昇トレンド (SMA20上向き ＆ 株価がSMA20の上)"
                elif sma20_diff < 0 and current_price < sma20:
                    trend_msg = "🔴 下落トレンド (SMA20下向き ＆ 株価がSMA20の下)"
                else:
                    trend_msg = "🟡 ボックストレンド (方向感なし)"
                    
                st.info(trend_msg)
                
                st.markdown("**直近1週間の値動き (四本値)**")
                recent_5d = df.tail(5)[['Open', 'High', 'Low', 'Close']].round(1)
                recent_5d.index = recent_5d.index.strftime('%m/%d')
                st.table(recent_5d.T)
                
            with col2:
                st.subheader("🤖 AIによる1ヶ月後のトレンド予測")
                if prob_dict:
                    st.write("過去3年のデータから算出した、20営業日後の状態確率")
                    
                    p_up = prob_dict.get(1, 0) * 100
                    p_box = prob_dict.get(0, 0) * 100
                    p_down = prob_dict.get(-1, 0) * 100
                    
                    st.metric("🟢 上昇トレンドにいる確率", f"{p_up:.1f}%")
                    st.metric("🟡 ボックストレンドにいる確率", f"{p_box:.1f}%")
                    st.metric("🔴 下落トレンドにいる確率", f"{p_down:.1f}%")
                    
                    # プログレスバー風の視覚化
                    st.progress(prob_dict.get(1, 0), text="上昇")
                    st.progress(prob_dict.get(-1, 0), text="下落")
                else:
                    st.warning("データ不足のためAI予測を実行できません。")

            st.markdown("---")
            st.subheader("Step 2 & 3: 直近で点灯中のシグナル")
            
            # 直近20日で点灯したシグナルを抽出するヘルパー関数
            def get_recent_signals(df, col_name, days=20):
                recent = df[col_name].tail(days)
                if recent.any():
                    date = recent[recent == True].index[-1].strftime('%Y-%m-%d')
                    return True, date
                return False, "-"

            buy_signals_map = {
                "グランビルの法則(買い①: 転換)": "Sig_Granville_Buy_1",
                "グランビルの法則(買い②: 押し目)": "Sig_Granville_Buy_2",
                "グランビルの法則(買い③: 乗せ)": "Sig_Granville_Buy_3",
                "ゴールデンクロス(5日/20日)": "Sig_GC",
                "パーフェクトオーダー": "Sig_PO_Buy",
                "新高値ブレイクアウト(半年)": "Sig_Breakout_Buy",
                "MACD上抜け": "Sig_MACD_Buy",
                "ダウ理論(上昇波)": "Sig_Dow_Buy",
                "一目均衡表(三役好転)": "Sig_Ichimoku_Buy"
            }
            
            sell_signals_map = {
                "グランビルの法則(売り①: 転換)": "Sig_Granville_Sell_1",
                "グランビルの法則(売り②: 戻り)": "Sig_Granville_Sell_2",
                "グランビルの法則(売り③: 乗せ)": "Sig_Granville_Sell_3",
                "デッドクロス(5日/20日)": "Sig_DC",
                "MACD下抜け": "Sig_MACD_Sell",
                "ダウ理論(下落波)": "Sig_Dow_Sell",
                "一目均衡表(三役逆転)": "Sig_Ichimoku_Sell",
                "ローソク足(包み足・陰線)": "Sig_Engulfing_Sell",
                "ローソク足(否定陰線)": "Sig_Negation_Sell"
            }
            
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                st.success("🟢 【買い】シグナル (直近20日)")
                for name, col in buy_signals_map.items():
                    act, date = get_recent_signals(df, col)
                    if act:
                        score = calculate_signal_score(df, col, True)
                        st.write(f"- **{name}** (点灯日: {date}) - 精度: **{score:.1f}点**")
                        
            with col_sell:
                st.error("🔴 【売り】シグナル (直近20日)")
                for name, col in sell_signals_map.items():
                    act, date = get_recent_signals(df, col)
                    if act:
                        score = calculate_signal_score(df, col, False)
                        st.write(f"- **{name}** (点灯日: {date}) - 精度: **{score:.1f}点**")

            st.markdown("---")
            with st.expander("📈 過去3年間のシグナル実績（平均変動割合）", expanded=False):
                st.write("過去3年間における各シグナル点灯後の平均株価変動率（%）と、直近の成績を重視した総合精度スコアです。")
                
                # 買い実績集計
                buy_rows = []
                for name, col in buy_signals_map.items():
                    score = calculate_signal_score(df, col, True)
                    idx_list = df.index[df[col] == True]
                    count = len(idx_list)
                    
                    r_1w, r_2w, r_3w, r_4w = [], [], [], []
                    for idx in idx_list:
                        try:
                            pos = df.index.get_loc(idx)
                            p = df['Close'].iloc[pos]
                            if pos+5 < len(df): r_1w.append((df['Close'].iloc[pos+5]-p)/p*100)
                            if pos+10 < len(df): r_2w.append((df['Close'].iloc[pos+10]-p)/p*100)
                            if pos+15 < len(df): r_3w.append((df['Close'].iloc[pos+15]-p)/p*100)
                            if pos+20 < len(df): r_4w.append((df['Close'].iloc[pos+20]-p)/p*100)
                        except: pass
                    
                    buy_rows.append({
                        "シグナル名": name,
                        "点灯回数": count,
                        "精度スコア": round(score, 1),
                        "1週後": f"{np.mean(r_1w):.1f}%" if r_1w else "-",
                        "2週後": f"{np.mean(r_2w):.1f}%" if r_2w else "-",
                        "3週後": f"{np.mean(r_3w):.1f}%" if r_3w else "-",
                        "4週後": f"{np.mean(r_4w):.1f}%" if r_4w else "-"
                    })
                
                st.subheader("🟢 買いシグナル実績")
                st.table(pd.DataFrame(buy_rows).set_index("シグナル名"))
                
                # 売り実績集計
                sell_rows = []
                for name, col in sell_signals_map.items():
                    score = calculate_signal_score(df, col, False)
                    idx_list = df.index[df[col] == True]
                    count = len(idx_list)
                    
                    r_1d, r_2d, r_3d, r_1w = [], [], [], []
                    for idx in idx_list:
                        try:
                            pos = df.index.get_loc(idx)
                            p = df['Close'].iloc[pos]
                            if pos+1 < len(df): r_1d.append((df['Close'].iloc[pos+1]-p)/p*100)
                            if pos+2 < len(df): r_2d.append((df['Close'].iloc[pos+2]-p)/p*100)
                            if pos+3 < len(df): r_3d.append((df['Close'].iloc[pos+3]-p)/p*100)
                            if pos+5 < len(df): r_1w.append((df['Close'].iloc[pos+5]-p)/p*100)
                        except: pass
                        
                    sell_rows.append({
                        "シグナル名": name,
                        "点灯回数": count,
                        "精度スコア": round(score, 1),
                        "1日後": f"{np.mean(r_1d):.1f}%" if r_1d else "-",
                        "2日後": f"{np.mean(r_2d):.1f}%" if r_2d else "-",
                        "3日後": f"{np.mean(r_3d):.1f}%" if r_3d else "-",
                        "1週後": f"{np.mean(r_1w):.1f}%" if r_1w else "-"
                    })
                
                st.subheader("🔴 売りシグナル実績")
                st.table(pd.DataFrame(sell_rows).set_index("シグナル名"))

            st.markdown("---")
            with st.expander("📊 ローソク足チャート ＆ シグナル発生ポイント (全期間)", expanded=True):
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # ローソク足
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                             low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                
                # 移動平均線
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='orange', width=1), name='SMA 5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1.5), name='SMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='green', width=1.5), name='SMA 60'), row=1, col=1)
                
                # 雲 (一目均衡表)
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', name='Cloud'), row=1, col=1)

                # マーカー描画用のヘルパー関数
                def add_markers(df, col_names, is_buy, is_dow=False):
                    x_dates = []
                    y_prices = []
                    texts = []
                    
                    for idx, row in df.iterrows():
                        active_sigs = []
                        for name, col in col_names.items():
                            if pd.notna(row.get(col)) and bool(row.get(col)):
                                active_sigs.append(name)
                        
                        if active_sigs:
                            x_dates.append(idx)
                            # 買いは安値の下、売りは高値の上に少しずらして配置
                            offset = row['Close'] * 0.02
                            y_prices.append(row['Low'] - offset if is_buy else row['High'] + offset)
                            texts.append("<br>".join(active_sigs))
                    
                    if x_dates:
                        if is_dow:
                            color = 'yellow' if is_buy else 'purple'
                            symbol = 'star'
                            size = 12
                            name_suffix = "ダウ理論"
                        else:
                            color = '#00BCD4' if is_buy else '#E91E63' # 買い:水色, 売り:ピンク
                            symbol = 'triangle-up' if is_buy else 'triangle-down'
                            size = 10
                            name_suffix = "一般"
                            
                        fig.add_trace(go.Scatter(
                            x=x_dates, y=y_prices, mode='markers',
                            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color='black')),
                            name=f'{"Buy" if is_buy else "Sell"} Signal ({name_suffix})',
                            text=texts, hoverinfo='text'
                        ), row=1, col=1)

                # 一般シグナルの描画 (ダウ以外)
                buy_general = {k:v for k,v in buy_signals_map.items() if "ダウ" not in k}
                sell_general = {k:v for k,v in sell_signals_map.items() if "ダウ" not in k}
                add_markers(df, buy_general, is_buy=True, is_dow=False)
                add_markers(df, sell_general, is_buy=False, is_dow=False)
                
                # ダウ理論の描画
                buy_dow = {k:v for k,v in buy_signals_map.items() if "ダウ" in k}
                sell_dow = {k:v for k,v in sell_signals_map.items() if "ダウ" in k}
                add_markers(df, buy_dow, is_buy=True, is_dow=True)
                add_markers(df, sell_dow, is_buy=False, is_dow=True)

                # MACD (下段)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='red', width=1), name='Signal'), row=2, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['MACD_Signal'], name='Histogram', marker_color='gray'), row=2, col=1)

                fig.update_layout(height=800, title_text=f"{selected_name} テクニカルチャート (過去3年)", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
