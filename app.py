import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(
    page_title="日本株AIレコメンドデモ",
    layout="wide" # トップ10表示で画面を広く使うためにwideに変更
)

# --- 免責事項 ---
st.warning("**免責事項**: 本アプリは学習・研究目的のプロトタイプです。予測結果に基づく投資行動についていかなる責任も負いません。投資は自己責任で行ってください。")

st.title("日本株 AIトレンド予測＆レコメンド")
st.write("過去の株価データからテクニカル指標を計算し、機械学習（ランダムフォレスト）を用いて翌日および1週間後（5営業日後）の株価が上昇する確率を予測します。")

# --- 分析対象の銘柄リスト ---
TARGET_STOCKS = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "8306.T": "三菱UFJ F.G",
    "7974.T": "任天堂",
    "9983.T": "ファーストリテイリング",
    "8035.T": "東京エレクトロン",
    "6861.T": "キーエンス",
    "4063.T": "信越化学工業",
    "6981.T": "村田製作所",
    "8058.T": "三菱商事",
    "8031.T": "三井物産",
    "9432.T": "日本電信電話(NTT)",
    "9433.T": "KDDI",
    "4502.T": "武田薬品工業",
    "4568.T": "第一三共",
    "6501.T": "日立製作所",
    "6752.T": "パナソニック HLDGS",
    "3382.T": "セブン&アイ HLDGS",
    "9020.T": "東日本旅客鉄道(JR東日本)",
    # --- 追加の代表的企業（21社） ---
    "6902.T": "デンソー",
    "6594.T": "ニデック",
    "4519.T": "中外製薬",
    "6098.T": "リクルートHLDGS",
    "8766.T": "東京海上HLDGS",
    "6367.T": "ダイキン工業",
    "8001.T": "伊藤忠商事",
    "9434.T": "ソフトバンク",
    "7741.T": "HOYA",
    "6954.T": "ファナック",
    "6503.T": "三菱電機",
    "4503.T": "アステラス製薬",
    "4661.T": "オリエンタルランド",
    "8053.T": "住友商事",
    "8316.T": "三井住友 F.G",
    "8411.T": "みずほ F.G",
    "2914.T": "日本たばこ産業",
    "4452.T": "花王",
    "5108.T": "ブリヂストン",
    "8801.T": "三井不動産",
    "8802.T": "三菱地所"
}

stock_count = len(TARGET_STOCKS)

# --- テクニカル指標の計算関数 ---
def add_technical_indicators(df):
    # 移動平均
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_25'] = df['Close'].rolling(window=25).mean()
    
    # 終値の変動率（リターン）
    df['Return'] = df['Close'].pct_change()
    
    # RSI (14日間)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ボラティリティ (5日間の標準偏差)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    
    # 前日比のボリューム変化率
    df['Vol_Change'] = df['Volume'].pct_change()

    return df

# --- データ取得とモデル学習関数（キャッシュして高速化） ---
@st.cache_data(ttl=3600) # 1時間キャッシュ
def analyze_stock(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 3) # 過去3年分
    
    # yfinanceでデータ取得
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(df) < 50:
        return None, None, None, None, None, None, None, None, None, None
    
    # カラムがマルチインデックスになる場合(yfのバージョンによる)の対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = add_technical_indicators(df)
    
    # 目的変数 (分類用・翌日): 翌日の終値が今日の終値より高ければ 1 (上昇), それ以外は 0 (下落)
    df['Target_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # 目的変数 (分類用・1週間後): 5営業日後の終値が今日の終値より高ければ 1, それ以外は 0
    df['Target_Class_1W'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    # 目的変数 (回帰用・終値): 翌日の終値そのもの
    df['Target_Reg'] = df['Close'].shift(-1)
    # 目的変数 (回帰用・始値): 翌日の始値そのもの
    df['Target_Reg_Open'] = df['Open'].shift(-1)
    
    # 欠損値を削除（最新の行=予測対象はTargetがNaNになるので別途保持）
    df_clean = df.dropna()
    
    # 特徴量 (Features)
    features = ['Close', 'Volume', 'SMA_5', 'SMA_25', 'Return', 'RSI', 'Volatility', 'Vol_Change']
    X = df_clean[features]
    y_class = df_clean['Target_Class']
    y_class_1w = df_clean['Target_Class_1W']
    y_reg = df_clean['Target_Reg']
    y_reg_open = df_clean['Target_Reg_Open']
    
    # 機械学習モデルの構築（分類: 翌日上がるか下がるか）
    model_class = RandomForestClassifier(n_estimators=100, random_state=42)
    model_class.fit(X, y_class)

    # 機械学習モデルの構築（分類: 1週間後上がるか下がるか）
    model_class_1w = RandomForestClassifier(n_estimators=100, random_state=42)
    model_class_1w.fit(X, y_class_1w)
    
    # 機械学習モデルの構築（回帰: 具体的な終値）
    model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reg.fit(X, y_reg)
    
    # 機械学習モデルの構築（回帰: 具体的な始値）
    model_reg_open = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reg_open.fit(X, y_reg_open)
    
    # 最新のデータ（今日）の特徴量を取得して明日・1週間後を予測
    latest_data = df.iloc[-1:][features]
    
    # 1つ前の営業日（昨日など）の特徴量を取得して最新営業日の結果を予測（答え合わせ用）
    prev_data = df.iloc[-2:-1][features]
    
    if latest_data.isnull().values.any() or prev_data.isnull().values.any():
         # NaNが含まれる場合は予測不可（直近データ不足など）
         return df, None, None, None, None, None, None, None, None, None
         
    # 上昇する確率（翌日）を取得
    prediction_proba = model_class.predict_proba(latest_data)[0][1] 
    prev_prediction_proba = model_class.predict_proba(prev_data)[0][1]

    # 上昇する確率（1週間後）を取得
    prediction_proba_1w = model_class_1w.predict_proba(latest_data)[0][1]
    prev_prediction_proba_1w = model_class_1w.predict_proba(prev_data)[0][1]
    
    # 翌日の終値を予測
    predicted_price = model_reg.predict(latest_data)[0]
    prev_predicted_price = model_reg.predict(prev_data)[0]
    
    # 翌日の始値を予測
    predicted_open = model_reg_open.predict(latest_data)[0]
    prev_predicted_open = model_reg_open.predict(prev_data)[0]
    
    # 実際の答え合わせ（最新営業日の終値 > 1つ前の営業日の終値）
    latest_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    actual_up = latest_close > prev_close
    
    return df, prediction_proba, prev_prediction_proba, actual_up, predicted_price, prev_predicted_price, predicted_open, prev_predicted_open, prediction_proba_1w, prev_prediction_proba_1w

# --- メイン処理 ---
if st.button(f"分析を開始する（対象{stock_count}社: 約30〜60秒かかります）"):
    with st.spinner(f"対象{stock_count}社の株価データを取得し、AIモデルで分析中..."):
        results = []
        progress_bar = st.progress(0)
        
        for i, (ticker, name) in enumerate(TARGET_STOCKS.items()):
            df, proba, prev_proba, actual_up, predicted_price, prev_predicted_price, predicted_open, prev_predicted_open, proba_1w, prev_proba_1w = analyze_stock(ticker)
            if proba is not None:
                current_price = df['Close'].iloc[-1]
                current_open = df['Open'].iloc[-1]
                # Seriesや配列になっている場合の対応
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
                if isinstance(current_open, pd.Series):
                    current_open = current_open.iloc[0]
                
                # 対象日付の計算
                latest_date_dt = df.index[-1]
                next_date_dt = latest_date_dt + pd.offsets.BDay(1)
                prev_date_dt = df.index[-2]
                week_later_dt = latest_date_dt + pd.offsets.BDay(5)
                
                latest_date_str = latest_date_dt.strftime('%Y/%m/%d')
                next_date_str = next_date_dt.strftime('%Y/%m/%d')
                prev_date_str = prev_date_dt.strftime('%Y/%m/%d')
                week_later_str = week_later_dt.strftime('%Y/%m/%d')
                    
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": float(current_price),
                    "open": float(current_open),
                    "latest_date": latest_date_str,
                    "next_date": next_date_str,
                    "prev_date": prev_date_str,
                    "week_later_date": week_later_str,
                    "score": proba * 100, # 翌日上昇確率
                    "prev_score": prev_proba * 100, 
                    "score_1w": proba_1w * 100, # 1週間後上昇確率
                    "prev_score_1w": prev_proba_1w * 100,
                    "actual_up": actual_up, # 実際の値動き
                    "predicted_price": float(predicted_price), # 予測終値
                    "prev_predicted_price": float(prev_predicted_price), 
                    "predicted_open": float(predicted_open), # 予測始値
                    "prev_predicted_open": float(prev_predicted_open), 
                    "df": df
                })
            progress_bar.progress((i + 1) / stock_count)
            
        # スコア（翌日上昇確率）が高い順にソート
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        st.session_state['analysis_results'] = results_sorted
        st.success(f"{len(results)}社の分析が完了しました！")

# --- 結果の表示 ---
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    # --- ランキング切り替えUI ---
    sort_basis = st.radio(
        "ランキングの基準を選択:",
        ("翌日予測ベース", "1週間後予測ベース"),
        horizontal=True
    )
    
    if sort_basis == "翌日予測ベース":
        sort_key = 'score'
    else:
        sort_key = 'score_1w'
        
    results_sorted = sorted(results, key=lambda x: x[sort_key], reverse=True)
    
    col_up, col_down = st.columns(2)
    
    with col_up:
        st.subheader(f"上昇期待度 トップ10 ({sort_basis})")
        # スコアが高い順（最大10件）
        for i in range(min(10, len(results_sorted))):
            stock = results_sorted[i]
            with st.container(border=True):
                st.markdown(f"**第{i+1}位: {stock['name']}**")
                st.caption(f"{stock['ticker']} | {stock['latest_date']} 終値: ¥{stock['price']:,.0f} (始値: ¥{stock['open']:,.0f})")
                if sort_basis == "翌日予測ベース":
                    st.markdown(f"<h3 style='color: green; margin-top: -10px; margin-bottom: 0px;'>翌日上昇確率: {stock['score']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>1週間後({stock['week_later_date']})上昇確率: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: green; margin-top: -10px; margin-bottom: 0px;'>1週間後({stock['week_later_date']})上昇確率: {stock['score_1w']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>翌日上昇確率: {stock['score']:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"**{stock['next_date']}** 予測終値: **¥{stock['predicted_price']:,.0f}** | 予測始値: **¥{stock['predicted_open']:,.0f}**", unsafe_allow_html=True)

    with col_down:
        st.subheader(f"下落警戒 ワースト10 ({sort_basis})")
        # スコアが低い順（後ろから取得、最大10件）
        for i in range(min(10, len(results_sorted))):
            stock = results_sorted[-(i+1)] # リストの後ろから取得
            down_prob = 100 - stock['score'] # 翌日下落確率
            down_prob_1w = 100 - stock['score_1w'] # 1週間後下落確率
            with st.container(border=True):
                st.markdown(f"**第{i+1}位: {stock['name']}**")
                st.caption(f"{stock['ticker']} | {stock['latest_date']} 終値: ¥{stock['price']:,.0f} (始値: ¥{stock['open']:,.0f})")
                if sort_basis == "翌日予測ベース":
                    st.markdown(f"<h3 style='color: red; margin-top: -10px; margin-bottom: 0px;'>翌日下落確率: {down_prob:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>1週間後({stock['week_later_date']})下落確率: {down_prob_1w:.1f}%</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: red; margin-top: -10px; margin-bottom: 0px;'>1週間後({stock['week_later_date']})下落確率: {down_prob_1w:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>翌日下落確率: {down_prob:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"**{stock['next_date']}** 予測終値: **¥{stock['predicted_price']:,.0f}** | 予測始値: **¥{stock['predicted_open']:,.0f}**", unsafe_allow_html=True)

    st.divider()
    
    # 個別チャートの表示セクション
    st.subheader("銘柄ごとの詳細チャート確認")
    selected_name = st.selectbox("詳細を見たい銘柄を選択してください", [r['name'] for r in sorted(results, key=lambda x: x['ticker'])]) # プルダウンはティッカー順に
    selected_stock = next(r for r in results if r['name'] == selected_name)
    df = selected_stock['df']
    
    # 直近半年分のデータをプロット
    plot_df = df.tail(120) 
    
    # Plotlyでローソク足チャートを描画
    fig = go.Figure(data=[go.Candlestick(x=plot_df.index,
                    open=plot_df['Open'].squeeze(),
                    high=plot_df['High'].squeeze(),
                    low=plot_df['Low'].squeeze(),
                    close=plot_df['Close'].squeeze(),
                    name='ローソク足')])
    
    # 移動平均線の追加
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_5'].squeeze(), line=dict(color='orange', width=1), name='5日移動平均'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_25'].squeeze(), line=dict(color='blue', width=1), name='25日移動平均'))
    
    fig.update_layout(
        title=f"{selected_name} ({selected_stock['ticker']}) の株価推移",
        yaxis_title="株価 (円)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0) # スマホ向けに余白を削る
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    down_score = 100 - selected_stock['score']
    down_score_1w = 100 - selected_stock['score_1w']
    st.info(f"**AIの分析 ({selected_name})**:\n\n"
            f"**【翌日 ({selected_stock['next_date']}) のトレンド】** 上がる確率: **{selected_stock['score']:.1f}%** / 下がる確率: **{down_score:.1f}%**\n\n"
            f"**【1週間後 ({selected_stock['week_later_date']}) のトレンド】** 上がる確率: **{selected_stock['score_1w']:.1f}%** / 下がる確率: **{down_score_1w:.1f}%**\n\n"
            f"**{selected_stock['next_date']} のAI予測終値: ¥{selected_stock['predicted_price']:,.0f}** ({selected_stock['latest_date']} 終値 ¥{selected_stock['price']:,.0f})\n\n"
            f"**{selected_stock['next_date']} のAI予測始値: ¥{selected_stock['predicted_open']:,.0f}** ({selected_stock['latest_date']} 始値 ¥{selected_stock['open']:,.0f})")
    
    # --- 直近の予測の答え合わせ表示 ---
    st.markdown("### 直近の予測の答え合わせ（翌日予測）")
    st.write(f"1つ前の営業日（{selected_stock['prev_date']}）のデータを使って、最新営業日（{selected_stock['latest_date']}）の株価変動を予測した結果です。")
    
    if 'prev_score' in selected_stock:
        prev_score = selected_stock['prev_score']
        actual_result_text = "上昇" if selected_stock['actual_up'] else "下落/変わらず"
        prediction_text = "上昇" if prev_score > 50 else "下落"
        is_correct = (prev_score > 50) == selected_stock['actual_up']
        
        prev_pred_price = selected_stock['prev_predicted_price']
        prev_pred_open = selected_stock['prev_predicted_open']
        
        latest_actual_price = selected_stock['price']
        latest_actual_open = selected_stock['open']
        
        price_diff = latest_actual_price - prev_pred_price
        open_diff = latest_actual_open - prev_pred_open
        
        if is_correct:
            st.success(f"**方向的中！** \n\n予測方向: **{prediction_text}** (確率 {prev_score:.1f}%) ➔ 実際: **{actual_result_text}**")
        else:
            st.error(f"**方向ハズレ** \n\n予測方向: **{prediction_text}** (確率 {prev_score:.1f}%) ➔ 実際: **{actual_result_text}**")
            
        st.write(f"**価格の答え合わせ ({selected_stock['latest_date']}):**")
        st.write(f"- 【終値】AI予測: **¥{prev_pred_price:,.0f}** ➔ 実際: **¥{latest_actual_price:,.0f}** (誤差: ¥{abs(price_diff):,.0f})")
        st.write(f"- 【始値】AI予測: **¥{prev_pred_open:,.0f}** ➔ 実際: **¥{latest_actual_open:,.0f}** (誤差: ¥{abs(open_diff):,.0f})")
    else:
        st.warning("新しい予測データが読み込まれていません。再度「分析を開始する」ボタンを押してデータを更新してください。")
