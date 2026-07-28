import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(
    page_title="日本株AIレコメンドエンジン",
    layout="wide"
)

st.title("日本株 AIトレンド予測＆レコメンドエンジン")
st.write("過去の株価データからテクニカル指標を計算し、機械学習（ランダムフォレスト）を用いて、あなたが設定したエグジットルールを達成する確率を予測します。")

# --- 分析対象の銘柄リスト ---
TARGET_STOCKS = {
    "1332.T": "ニッスイ", "1333.T": "マルハニチロ", "1414.T": "ショーボンド", "1605.T": "INPEX", 
    "1719.T": "安藤・間", "1721.T": "コムシス", "1801.T": "大成建設", "1802.T": "大林組", 
    "1803.T": "清水建設", "1808.T": "長谷工", "1812.T": "鹿島", "1878.T": "大東建託", 
    "1911.T": "住友林業", "1925.T": "大和ハウス", "1928.T": "積水ハウス", "1944.T": "きんでん", 
    "1951.T": "エクシオ", "1959.T": "九電工", "2127.T": "日本M&A", "2175.T": "エス・エム・エス", 
    "2181.T": "パーソル", "2201.T": "森永製菓", "2212.T": "山崎製パン", "2229.T": "カルビー", 
    "2267.T": "ヤクルト", "2269.T": "明治HLDGS", "2282.T": "日本ハム", "2331.T": "ALSOK", 
    "2371.T": "カカクコム", "2413.T": "エムスリー", "2432.T": "DeNA", "2501.T": "サッポロ", 
    "2502.T": "アサヒ", "2503.T": "キリン", "2587.T": "サントリーBF", "2670.T": "ABCマート", 
    "2768.T": "双日", "2801.T": "キッコーマン", "2802.T": "味の素", "2809.T": "キユーピー", 
    "2871.T": "ニチレイ", "2897.T": "日清食品", "2914.T": "JT", "3003.T": "ヒューリック", 
    "3038.T": "神戸物産", "3064.T": "モノタロウ", "3086.T": "J.フロント", "3088.T": "マツキヨココカラ", 
    "3092.T": "ZOZO", "3099.T": "三越伊勢丹", "3116.T": "トヨタ紡織", "3132.T": "マクニカ", 
    "3141.T": "ウエルシア", "3182.T": "オイシックス", "3197.T": "すかいらーく", "3231.T": "野村不動産", 
    "3288.T": "オープンハウス", "3289.T": "東急不動産", "3291.T": "飯田GHD", "3349.T": "コスモス薬品", 
    "3382.T": "セブン&アイ", "3391.T": "ツルハ", "3401.T": "帝人", "3402.T": "東レ", 
    "3405.T": "クラレ", "3407.T": "旭化成", "3436.T": "SUMCO", "3626.T": "TIS", 
    "3632.T": "グリー", "3659.T": "ネクソン", "3697.T": "SHIFT", "3762.T": "テクマトリックス", 
    "3765.T": "ガンホー", "3769.T": "GMOペイメント", "3774.T": "IIJ", "3861.T": "王子HLDGS", 
    "3863.T": "日本製紙", "3903.T": "gumi", "3923.T": "ラクス", "3941.T": "レンゴー", 
    "4004.T": "レゾナック", "4005.T": "住友化学", "4021.T": "日産化学", "4041.T": "日本曹達", 
    "4042.T": "東ソー", "4043.T": "トクヤマ", "4061.T": "デンカ", "4062.T": "イビデン", 
    "4063.T": "信越化学", "4088.T": "エア・ウォーター", "4091.T": "日本酸素", "4114.T": "日本触媒", 
    "4118.T": "カネカ", "4151.T": "協和キリン", "4182.T": "三菱ガス化学", "4183.T": "三井化学", 
    "4185.T": "JSR", "4188.T": "三菱ケミカル", "4202.T": "ダイセル", "4204.T": "積水化学", 
    "4208.T": "UBE", "4307.T": "NRI", "4324.T": "電通", "4348.T": "インフォコム", 
    "4401.T": "ADEKA", "4443.T": "Sansan", "4452.T": "花王", "4502.T": "武田薬品", 
    "4503.T": "アステラス", "4506.T": "住友ファーマ", "4507.T": "塩野義", "4516.T": "日本新薬", 
    "4519.T": "中外製薬", "4521.T": "科研製薬", "4523.T": "エーザイ", "4527.T": "ロート製薬", 
    "4528.T": "小野薬品", "4530.T": "久光製薬", "4536.T": "参天製薬", "4543.T": "テルモ", 
    "4544.T": "HUグループ", "4552.T": "JCRファーマ", "4568.T": "第一三共", "4578.T": "大塚HLDGS", 
    "4581.T": "大正製薬", "4612.T": "日本ペイント", "4613.T": "関西ペイント", "4631.T": "DIC", 
    "4661.T": "オリエンタルランド", "4665.T": "ダスキン", "4666.T": "パーク24", "4684.T": "オービック", 
    "4689.T": "LINEヤフー", "4704.T": "トレンドマイクロ", "4716.T": "日本オラクル", "4732.T": "USS", 
    "4739.T": "伊藤忠テクノ", "4768.T": "大塚商会", "4816.T": "東映アニメ", "4901.T": "富士フイルム", 
    "4902.T": "コニカミノルタ", "4911.T": "資生堂", "4912.T": "ライオン", "4919.T": "ミルボン", 
    "4921.T": "ファンケル", "4922.T": "コーセー", "4927.T": "ポーラ・オルビス", "4967.T": "小林製薬", 
    "4980.T": "デクセリアルズ", "5019.T": "出光興産", "5020.T": "ENEOS", "5021.T": "コスモエネルギー", 
    "5101.T": "横浜ゴム", "5105.T": "TOYO TIRE", "5108.T": "ブリヂストン", "5110.T": "住友ゴム", 
    "5201.T": "AGC", "5214.T": "日本電気硝子", "5232.T": "住友大阪セメント", "5233.T": "太平洋セメント", 
    "5301.T": "東海カーボン", "5332.T": "TOTO", "5333.T": "日本碍子", "5334.T": "日本特殊陶業", 
    "5401.T": "日本製鉄", "5406.T": "神戸製鋼所", "5411.T": "JFE", "5471.T": "大同特殊鋼", 
    "5486.T": "日立金属", "5541.T": "大平洋金属", "5703.T": "日軽金", "5706.T": "三井金属", 
    "5711.T": "三菱マテリアル", "5713.T": "住友金属鉱山", "5714.T": "DOWA", "5726.T": "大阪チタニウム", 
    "5801.T": "古河電工", "5802.T": "住友電工", "5803.T": "フジクラ", "5831.T": "しずおかFG", 
    "5838.T": "楽天銀行", "5901.T": "東洋製罐", "5929.T": "三和HLDGS", "5938.T": "LIXIL", 
    "5947.T": "リンナイ", "5991.T": "ニッパツ", "6005.T": "三浦工業", "6013.T": "タクマ", 
    "6028.T": "テクノプロ", "6098.T": "リクルート", "6103.T": "オークマ", "6113.T": "アマダ", 
    "6141.T": "DMG森精機", "6146.T": "ディスコ", "6201.T": "豊田自動織機", "6268.T": "ナブテスコ", 
    "6273.T": "SMC", "6301.T": "コマツ", "6302.T": "住友重機械", "6305.T": "日立建機", 
    "6326.T": "クボタ", "6361.T": "荏原製作所", "6367.T": "ダイキン", "6370.T": "栗田工業", 
    "6383.T": "ダイフク", "6395.T": "タダノ", "6406.T": "フジテック", "6417.T": "SANKYO", 
    "6436.T": "アマノ", "6448.T": "ブラザー工業", "6465.T": "ホシザキ", "6471.T": "日本精工", 
    "6472.T": "NTN", "6473.T": "ジェイテクト", "6479.T": "ミネベアミツミ", "6481.T": "THK", 
    "6501.T": "日立製作所", "6503.T": "三菱電機", "6504.T": "富士電機", "6506.T": "安川電機", 
    "6508.T": "明電舎", "6526.T": "ソシオネクスト", "6532.T": "ベイカレント", "6586.T": "マキタ", 
    "6594.T": "ニデック", "6645.T": "オムロン", "6674.T": "GSユアサ", "6701.T": "NEC", 
    "6702.T": "富士通", "6723.T": "ルネサス", "6724.T": "セイコーエプソン", "6752.T": "パナソニック", 
    "6753.T": "シャープ", "6758.T": "ソニーG", "6762.T": "TDK", "6770.T": "アルプスアルパイン", 
    "6806.T": "ヒロセ電機", "6841.T": "横河電機", "6845.T": "アズビル", "6857.T": "アドバンテスト", 
    "6861.T": "キーエンス", "6869.T": "シスメックス", "6902.T": "デンソー", "6920.T": "レーザーテック", 
    "6923.T": "スタンレー電気", "6952.T": "カシオ", "6954.T": "ファナック", "6965.T": "浜松ホトニクス", 
    "6971.T": "京セラ", "6976.T": "太陽誘電", "6981.T": "村田製作所", "6988.T": "日東電工", 
    "7011.T": "三菱重工業", "7012.T": "川崎重工業", "7013.T": "IHI", "7180.T": "九州FG", 
    "7181.T": "かんぽ生命", "7182.T": "ゆうちょ銀行", "7186.T": "コンコルディア", "7189.T": "西日本FH", 
    "7201.T": "日産自動車", "7202.T": "いすゞ自動車", "7203.T": "トヨタ自動車", "7205.T": "日野自動車", 
    "7211.T": "三菱自動車", "7239.T": "タチエス", "7240.T": "NOK", "7259.T": "アイシン", 
    "7261.T": "マツダ", "7267.T": "ホンダ", "7269.T": "スズキ", "7270.T": "SUBARU", 
    "7272.T": "ヤマハ発動機", "7282.T": "豊田合成", "7309.T": "シマノ", "7381.T": "北國FHD", 
    "7453.T": "良品計画", "7459.T": "メディパル", "7518.T": "ネットワン", "7532.T": "パンパシフィック", 
    "7550.T": "ゼンショー", "7649.T": "スギHD", "7701.T": "島津製作所", "7718.T": "スター精密", 
    "7729.T": "東京精密", "7731.T": "ニコン", "7733.T": "オリンパス", "7735.T": "SCREEN", 
    "7741.T": "HOYA", "7747.T": "朝日インテック", "7751.T": "キヤノン", "7752.T": "リコー", 
    "7762.T": "シチズン時計", "7780.T": "メニコン", "7832.T": "バンダイナムコ", "7911.T": "TOPPAN", 
    "7912.T": "大日本印刷", "7936.T": "アシックス", "7951.T": "ヤマハ", "7974.T": "任天堂", 
    "8001.T": "伊藤忠", "8002.T": "丸紅", "8015.T": "豊田通商", "8020.T": "兼松", 
    "8031.T": "三井物産", "8035.T": "東京エレクトロン", "8053.T": "住友商事", "8058.T": "三菱商事", 
    "8113.T": "ユニ・チャーム", "8136.T": "サンリオ", "8227.T": "しまむら", "8233.T": "高島屋", 
    "8242.T": "H2Oリテイリング", "8252.T": "丸井グループ", "8282.T": "ケーズHD", "8303.T": "SBI新生銀行", 
    "8304.T": "あおぞら銀行", "8306.T": "三菱UFJ", "8308.T": "りそな", "8309.T": "三井住友トラスト", 
    "8316.T": "三井住友FG", "8331.T": "千葉銀行", "8354.T": "ふくおかFG", "8355.T": "静岡銀行", 
    "8377.T": "ほくほくFG", "8381.T": "山陰合同銀行", "8411.T": "みずほ", "8473.T": "SBI", 
    "8584.T": "ジャックス", "8591.T": "オリックス", "8593.T": "三菱HCキャピタル", "8601.T": "大和証券", 
    "8604.T": "野村HD", "8628.T": "松井証券", "8630.T": "SOMPO", "8697.T": "日本取引所", 
    "8725.T": "MS&AD", "8750.T": "第一生命", "8766.T": "東京海上", "8795.T": "T&D", 
    "8801.T": "三井不動産", "8802.T": "三菱地所", "8804.T": "東京建物", "8830.T": "住友不動産", 
    "8876.T": "リログループ", "8892.T": "日本エスコン", "8905.T": "イオンモール", "8919.T": "カチタス", 
    "9001.T": "東武鉄道", "9005.T": "東急", "9007.T": "小田急電鉄", "9008.T": "京王電鉄", 
    "9009.T": "京成電鉄", "9020.T": "JR東日本", "9021.T": "JR西日本", "9022.T": "JR東海", 
    "9024.T": "西武HD", "9041.T": "近鉄GHD", "9042.T": "阪急阪神", "9045.T": "京阪HD", 
    "9048.T": "名古屋鉄道", "9062.T": "日本通運", "9064.T": "ヤマトHD", "9065.T": "山九", 
    "9069.T": "センコーGHD", "9089.T": "SGホールディングス", "9101.T": "日本郵船", "9104.T": "商船三井", 
    "9107.T": "川崎汽船", "9142.T": "JR九州", "9147.T": "NIPPON EXPRESS", 
    "9201.T": "日本航空", "9202.T": "ANA", "9301.T": "三菱倉庫", "9404.T": "日テレHD", 
    "9409.T": "テレビ朝日HD", "9412.T": "スカパーJ", "9432.T": "NTT", "9433.T": "KDDI", 
    "9434.T": "ソフトバンク", "9501.T": "東京電力", "9502.T": "中部電力", "9503.T": "関西電力", 
    "9504.T": "中国電力", "9505.T": "北陸電力", "9506.T": "東北電力", "9507.T": "四国電力", 
    "9508.T": "九州電力", "9509.T": "北海道電力", "9513.T": "電源開発", "9531.T": "東京瓦斯", 
    "9532.T": "大阪瓦斯", "9533.T": "東邦瓦斯", "9602.T": "東宝", "9613.T": "NTTデータ", 
    "9627.T": "アインHD", "9684.T": "スクウェア・エニックス", "9697.T": "カプコン", "9704.T": "アゴーラ", 
    "9719.T": "SCSK", "9735.T": "セコム", "9766.T": "コナミ", "9831.T": "ヤマダHD", 
    "9843.T": "ニトリ", "9962.T": "ミスミG", "9983.T": "ファーストリテイリング", "9984.T": "ソフトバンクG", 
    "9989.T": "サンドラッグ", "9990.T": "サックスバー"
}

stock_count = len(TARGET_STOCKS)

# --- テクニカル指標の計算関数 ---
def add_technical_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_25'] = df['Close'].rolling(window=25).mean()
    df['SMA_75'] = df['Close'].rolling(window=75).mean() # 追加: 中長期トレンド
    df['Return'] = df['Close'].pct_change()
    
    # MACDの追加 (スイングトレードのモメンタム指標)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ボリンジャーバンド幅の追加 (ボラティリティの収縮・拡大を捉え、急騰を予測)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Std'] * 4) / df['BB_Mid']
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['Vol_Change'] = df['Volume'].pct_change()

    return df

# --- モデル学習関数 ---
def analyze_stock_data(df, n_estimators=100, holding_period=5, profit_target_pct=10.0, stop_loss_pct=-5.0):
    if len(df) < 100: # 75日線を計算するため、足切りを100日に変更
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = add_technical_indicators(df)
    
    # --- 実践的なエグジットルールに基づく目的変数の作成 ---
    
    target_class = []
    
    for i in range(len(df)):
        if i >= len(df) - holding_period:
            target_class.append(np.nan)
            continue
            
        current_close = df['Close'].iloc[i]
        success = 0 # デフォルトは失敗(0)
        
        # 指定期間の動きをシミュレーション
        for j in range(1, holding_period + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]
            
            # 現在値に対する変動率を計算
            high_pct = (future_high - current_close) / current_close * 100 if current_close > 0 else 0
            low_pct = (future_low - current_close) / current_close * 100 if current_close > 0 else 0
            
            # ルール判定
            if low_pct <= stop_loss_pct:
                # 損切りに触れた場合、失敗(0)としてループを抜ける
                break 
            elif high_pct >= profit_target_pct:
                # 損切りに触れる前に利確に到達した場合、成功(1)としてループを抜ける
                success = 1
                break
                
        target_class.append(success)
        
    df['Target_Class_Rule'] = target_class
    
    # 無限大(inf)をNaNに変換してから削除（出来高0などで発生するエラーを回避）
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 特徴量リストを更新
    features = ['Close', 'Volume', 'SMA_5', 'SMA_25', 'SMA_75', 'Return', 'RSI', 'Volatility', 'Vol_Change', 'MACD', 'MACD_Signal', 'BB_Width']
    
    # 目的変数がNaNの行（直近5日分など）を削除し、学習データを作成
    df_clean = df.dropna(subset=['Target_Class_Rule'] + features)
    
    X = df_clean[features]
    
    # 1クラスしかない場合（例：過去すべて失敗）はエラーになるため回避
    if len(df_clean['Target_Class_Rule'].unique()) < 2:
        return None
    
    # モデルの構築と学習 (class_weight='balanced'を追加し、AIの怠慢を防止)
    model_class = RandomForestClassifier(
        n_estimators=n_estimators, 
        class_weight='balanced', 
        random_state=42
    ).fit(X, df_clean['Target_Class_Rule'])
    
    # 予測は「直近の最新データ」に対して行う
    latest_data = df.iloc[-1:][features]
    
    if latest_data.isnull().values.any():
         return None
         
    # 予測の実行 (条件をクリアする確率)
    prediction_proba = model_class.predict_proba(latest_data)[0][1]
    
    return {
        "df": df,
        "proba_rule": prediction_proba
    }

# --- 全銘柄の分析処理を一括キャッシュ（24時間保持） ---
@st.cache_data(ttl=86400, show_spinner=False)
def generate_all_results(years=3.0, n_estimators=100, top_n=134, holding_period=5, profit_target_pct=10.0, stop_loss_pct=-5.0):
    results = []
    progress_bar = st.progress(0)
    time_text = st.empty()  # 残り時間表示用の領域を作成
    
    tickers = list(TARGET_STOCKS.keys())
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(365 * years))
    
    # 全銘柄のデータを一括ダウンロード（通信回数を1回に減らし高速化）
    df_all = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True, progress=False)
    
    # --- 売買代金の計算と上位抽出 ---
    trading_values = {}
    valid_tickers = []
    
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                df_ticker = df_all[ticker].copy()
            else:
                df_ticker = df_all.copy()
            df_ticker.dropna(how='all', inplace=True)
            
            if len(df_ticker) > 0:
                def get_val(val):
                    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                
                latest_close = get_val(df_ticker['Close'].iloc[-1])
                latest_volume = get_val(df_ticker['Volume'].iloc[-1])
                trading_value = latest_close * latest_volume
                
                trading_values[ticker] = {
                    'value': trading_value,
                    'df': df_ticker
                }
                valid_tickers.append(ticker)
        except Exception:
            pass
            
    # 売買代金が大きい順にソートし、上位 top_n 社を抽出
    sorted_tickers = sorted(valid_tickers, key=lambda x: trading_values[x]['value'], reverse=True)[:top_n]
    
    # --- 抽出された上位企業のみを解析 ---
    start_time = time.time()
    total_tickers = len(sorted_tickers)
    
    for i, ticker in enumerate(sorted_tickers):
        name = TARGET_STOCKS[ticker]
        df_ticker = trading_values[ticker]['df']
            
        # 木の数を渡して分析実行
        analysis = analyze_stock_data(df_ticker, n_estimators=n_estimators, holding_period=holding_period, profit_target_pct=profit_target_pct, stop_loss_pct=stop_loss_pct)
        
        if analysis is not None:
            df = analysis["df"]
            
            def get_val(val):
                return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

            current_price = get_val(df['Close'].iloc[-1])
            current_open = get_val(df['Open'].iloc[-1])
            current_high = get_val(df['High'].iloc[-1])
            current_low = get_val(df['Low'].iloc[-1])
            
            current_sma_5 = get_val(df['SMA_5'].iloc[-1])
            current_sma_25 = get_val(df['SMA_25'].iloc[-1])
            current_sma_75 = get_val(df['SMA_75'].iloc[-1])
            current_macd = get_val(df['MACD'].iloc[-1])
            current_macd_signal = get_val(df['MACD_Signal'].iloc[-1])
            current_bb_mid = get_val(df['BB_Mid'].iloc[-1])
            current_bb_std = get_val(df['BB_Std'].iloc[-1])
            
            current_volume = get_val(df['Volume'].iloc[-1])
            vol_change = get_val(df['Vol_Change'].iloc[-1])
            vol_5d_avg = get_val(df['Volume'].rolling(window=5).mean().iloc[-1])
            current_rsi = get_val(df['RSI'].iloc[-1])
            
            # --- 推奨サイン（ポジティブ）判定用の追加 ---
            current_macd = get_val(df['MACD'].iloc[-1])
            current_macd_signal = get_val(df['MACD_Signal'].iloc[-1])
            
            # 対象日付の計算
            latest_date_dt = df.index[-1]
            week_later_dt = latest_date_dt + pd.offsets.BDay(holding_period)
            
            results.append({
                "ticker": ticker,
                "name": name,
                "price": current_price,
                "open": current_open,
                "high": current_high,
                "low": current_low,
                "current_sma_5": current_sma_5,
                "current_sma_25": current_sma_25,
                "current_sma_75": current_sma_75,
                "current_macd": current_macd,
                "current_macd_signal": current_macd_signal,
                "current_bb_mid": current_bb_mid,
                "current_bb_std": current_bb_std,
                "current_volume": current_volume,
                "vol_change": vol_change,
                "vol_5d_avg": vol_5d_avg,
                "current_rsi": current_rsi,
                "latest_date": latest_date_dt.strftime('%Y/%m/%d'),
                "week_later_date": week_later_dt.strftime('%Y/%m/%d'),
                "score_rule": analysis["proba_rule"] * 100,
                "df": df
            })
            
        # 進捗と残り時間の計算と表示
        processed_count = i + 1
        elapsed_time = time.time() - start_time
        avg_time_per_ticker = elapsed_time / processed_count
        est_remaining_time = avg_time_per_ticker * (total_tickers - processed_count)
        
        time_text.text(f"AIモデル学習中... 残り時間: 約 {int(est_remaining_time)} 秒 ({processed_count}/{total_tickers}社完了)")
        progress_bar.progress(processed_count / total_tickers)
        
    time_text.empty()
    progress_bar.empty()
    return results

# --- メイン処理 ---
st.markdown("### エグジットルール設定 (AIの学習目標)")
st.write("この条件を満たす確率をAIが学習・予測します。ルールを変更するとAIがゼロから学習し直します。")
col_rule1, col_rule2, col_rule3 = st.columns(3)
with col_rule1:
    holding_period = st.number_input("判定期間（営業日）", min_value=1, max_value=60, value=15, step=1, help="何日以内に条件を達成するか設定します。")
with col_rule2:
    profit_target_pct = st.number_input("利確ライン（%）", min_value=1.0, max_value=100.0, value=8.0, step=1.0, help="株価が何%上昇したら利益確定するか設定します。")
with col_rule3:
    stop_loss_pct = st.number_input("損切りライン（%）", min_value=-50.0, max_value=-1.0, value=-5.0, step=1.0, help="株価が何%下落したら損切りするか設定します（マイナスで入力）。")

st.markdown("### 分析詳細設定")
col_setting1, col_setting2 = st.columns(2)
with col_setting1:
    top_n = st.slider(f"解析対象とする売買代金上位の企業数を選択 (最大{stock_count}社)", min_value=10, max_value=stock_count, value=150, step=10, help="全銘柄から直近の売買代金が多い企業を自動選出し、AI解析の対象を絞ります。数を減らすと計算時間が短縮されます。")
with col_setting2:
    n_estimators = st.slider("AIモデルの木の本数 (10〜500本)", min_value=10, max_value=500, value=100, step=10, help="本数を増やすと予測が安定しますが、計算時間が長くなります。じっくり詳細分析したい時に500本をお試しください。")

col1, col2 = st.columns([1, 2])

with col1:
    run_btn = st.button("AI分析を実行 / 結果を表示", type="primary", help="設定されたルール・企業数・木の本数で詳細分析を行います。")
with col2:
    clear_btn = st.button("データを更新して再計算 (キャッシュクリア)")

if clear_btn:
    st.cache_data.clear()
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    st.success("キャッシュをクリアしました。「AI分析を実行」ボタンを押してください。")
    st.rerun()

# 実行ボタンが押された場合の処理
if run_btn:
    years = 3.0
    
    with st.spinner(f"対象全社のデータを取得し、売買代金上位{top_n}社をAIモデルで詳細分析中（木の本数: {n_estimators}本）...\n（計算済みの場合は一瞬で表示されます）"):
        results = generate_all_results(years=years, n_estimators=n_estimators, top_n=top_n, holding_period=holding_period, profit_target_pct=profit_target_pct, stop_loss_pct=stop_loss_pct)
        results_sorted = sorted(results, key=lambda x: x['score_rule'], reverse=True)
        st.session_state['analysis_results'] = results_sorted
        st.success(f"売買代金上位 {len(results)}社の詳細分析が完了（またはキャッシュから取得）しました！")

# --- 結果の表示 ---
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    # --- 厳選AIレコメンドセクション ---
    st.subheader("厳選AIレコメンド銘柄 (ルールベース特化)")
    
    max_price = st.number_input("予算上限：1株あたりの価格（円）を設定してください", min_value=100, max_value=150000, value=5000, step=100, help="指定した金額以下の銘柄のみをレコメンドします。（例: 5000円 = 100株単位で50万円）")
    
    st.write("設定されたトレードルールを満たし、かつご指定の予算内に収まる有望銘柄をピックアップします。")
    st.caption(f"【選定条件】・1株{max_price:,}円以下 ・75日移動平均線上(上昇トレンド) ・{holding_period}日以内に「{stop_loss_pct}%損切り」に触れず「+{profit_target_pct}%利確」を達成する確率が50%超")
    
    with st.expander("💡 【備忘録】表示されるテクニカルサインの読み解き方と活用法"):
        st.markdown("""
        各テクニカル指標には得意な相場と「ダマシ」のリスクがあります。単独で判断せず、**AIが算出した条件達成確率を主軸**とし、サインはサブ情報として活用するのが効果的です。

        **🔵 推奨サイン (背中を押すポジティブ要素)**
        *   **MACD上抜け(GC圏):** トレンドの勢いが上向き（ゴールデンクロス圏）です。AIの高勝率と重なれば、強い上昇の初動に乗れる可能性があります。
        *   **出来高急増:** 当日の出来高が過去5日平均の1.5倍以上に急増しています。大口（機関投資家など）の資金流入のサインであり、強いトレンドが長続きしやすいです。
        *   **短期上昇トレンド:** 株価が5日線の上にあり、かつ5日線が25日線上にある状態です。安定した上昇の波に乗っていることを示します。

        **🔴 注意サイン (エントリー見送り・出口準備のネガティブ要素)**
        *   **25日線割れ (優先度【中】):** 中期的な調整局面を示します。AIの勝率が高くても、再度25日線を上抜けるまで様子を見るのが安全です。
        *   **MACD下落傾向 (優先度【中】):** トレンドの勢いが下向きです。この下落傾向から「推奨サイン」に転じた瞬間が強いエントリー根拠になります。
        *   **5日線割れ (優先度【低】):** ノイズによるダマシが多いため、「短期的には少し弱含んでいる」程度の認識で十分です。
        *   **RSI過熱(70%超) / +2σ超過:** 強い上昇トレンドの最中はこれらが出続けます。「買うのをやめる」のではなく、**「そろそろ勢いが止まるかもしれないから、利益確定を早めにしよう」というエグジット警戒情報**として使います。
        """)
    
    recommended_stocks = []
    for stock in results:
        cond0 = stock['price'] <= max_price
        cond1 = stock['score_rule'] > 50
        cond2 = stock['price'] > stock['current_sma_75'] # トレンドフィルター: 75日線より上にあること
        
        if cond0 and cond1 and cond2:
            stock['recommend_score'] = stock['score_rule']
            recommended_stocks.append(stock)
            
    # 上限5個を撤廃し、全件を表示するよう変更
    recommended_stocks = sorted(recommended_stocks, key=lambda x: x['recommend_score'], reverse=True)
    
    if len(recommended_stocks) > 0:
        # 3列ごとに折り返して表示するレイアウト
        for i in range(0, len(recommended_stocks), 3):
            rec_cols = st.columns(3)
            for j in range(3):
                if i + j < len(recommended_stocks):
                    stock = recommended_stocks[i + j]
                    with rec_cols[j]:
                        with st.container(border=True):
                            st.markdown(f"**{stock['name']}**")
                            st.caption(f"{stock['ticker']}")
                            
                            # RSIの判定
                            if stock['current_rsi'] >= 70:
                                rsi_color = "red"
                                rsi_alert = " (過熱感)"
                            elif stock['current_rsi'] <= 30:
                                rsi_color = "green"
                                rsi_alert = " (売られすぎ)"
                            else:
                                rsi_color = "gray"
                                rsi_alert = ""
                            st.markdown(f"<p style='color: {rsi_color}; font-size: 0.85em; font-weight: bold; margin-bottom: 5px;'>RSI: {stock['current_rsi']:.1f}%{rsi_alert}</p>", unsafe_allow_html=True)

                            # --- 推奨サイン（ポジティブ）の判定と表示 ---
                            positive_alerts = []
                            if stock['current_macd'] > stock['current_macd_signal']:
                                positive_alerts.append("MACD上抜け(GC圏)")
                            if stock['current_volume'] >= (stock['vol_5d_avg'] * 1.5):
                                positive_alerts.append("出来高急増")
                            if stock['price'] > stock['current_sma_5'] and stock['current_sma_5'] > stock['current_sma_25']:
                                positive_alerts.append("短期上昇トレンド")
                                
                            if positive_alerts:
                                st.markdown(f"<p style='color: #1976D2; font-size: 0.8em; font-weight: bold; margin-top: -5px; margin-bottom: 5px; line-height: 1.2;'>🔵 推奨: {' / '.join(positive_alerts)}</p>", unsafe_allow_html=True)

                            # --- 注意サイン（ネガティブ）の判定と表示 ---
                            negative_alerts = []
                            if stock['price'] < stock['current_sma_5']:
                                negative_alerts.append("5日線割れ")
                            if stock['price'] < stock['current_sma_25']:
                                negative_alerts.append("25日線割れ")
                            if stock['current_volume'] < stock['vol_5d_avg']:
                                negative_alerts.append("出来高低迷")
                            if stock['current_macd'] < stock['current_macd_signal']:
                                negative_alerts.append("MACD下落傾向")
                            if stock['price'] >= stock['current_bb_mid'] + (stock['current_bb_std'] * 2):
                                negative_alerts.append("+2σ超過(過熱)")
                                
                            if negative_alerts:
                                st.markdown(f"<p style='color: #D32F2F; font-size: 0.8em; font-weight: bold; margin-top: -5px; margin-bottom: 5px; line-height: 1.2;'>🔴 注意: {' / '.join(negative_alerts)}</p>", unsafe_allow_html=True)

                            st.markdown(f"<p style='color: green; font-weight: bold; margin-bottom: 0px;'>条件達成確率: {stock['score_rule']:.1f}%</p>", unsafe_allow_html=True)
                            st.markdown("---")
                            
                            # 目標株価の計算
                            target_price = stock['price'] * (1 + profit_target_pct / 100)
                            stop_price = stock['price'] * (1 + stop_loss_pct / 100)
                            
                            st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px;'><b>直近 ({stock['latest_date']})</b><br>終値: ¥{stock['price']:,.0f} | 始値: ¥{stock['open']:,.0f}<br>高値: ¥{stock['high']:,.0f} | 安値: ¥{stock['low']:,.0f}</p>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size: 0.85em; margin-top: 5px; padding: 5px; border: 1px solid #ccc; border-radius: 5px;'><b>【AI予測の基準価格】</b><br>取得想定 (直近終値): <b>¥{stock['price']:,.0f}</b><br>🟢 利確目標 (+{profit_target_pct}%): <b>¥{target_price:,.0f}</b><br>🔴 損切目安 ({stop_loss_pct}%): <b>¥{stop_price:,.0f}</b></div>", unsafe_allow_html=True)
    else:
        st.info("現在、設定された厳しいルール条件（勝率50%超）を満たす銘柄はありません。相場環境が変わるのをお待ちください。")
        
    st.divider()
    
    st.subheader("銘柄ごとの詳細チャート確認 ＆ 保有銘柄の売りサイン診断")
    st.write("詳細を見たい銘柄、または保有中の銘柄を選択してください。現在のテクニカル指標から「売り（手仕舞い）」の警戒サインが出ていないか診断します。")
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_name = st.selectbox("銘柄を選択", [r['name'] for r in sorted(results, key=lambda x: x['ticker'])])
    with col_sel2:
        entry_price = st.number_input("取得単価（保有中の場合・任意）", min_value=0, value=0, step=100, help="実際に購入した株価を入力すると、その価格を基準にした目標株価を計算します。未入力(0)の場合は現在値を基準にします。")
        
    selected_stock = next(r for r in results if r['name'] == selected_name)
    df = selected_stock['df']
    
    # --- 売りサイン判定ロジック ---
    def get_val(val):
        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
        
    current_price = get_val(df['Close'].iloc[-1])
    current_open = get_val(df['Open'].iloc[-1])
    current_sma_5 = get_val(df['SMA_5'].iloc[-1])
    current_sma_25 = get_val(df['SMA_25'].iloc[-1])
    current_macd = get_val(df['MACD'].iloc[-1])
    current_macd_signal = get_val(df['MACD_Signal'].iloc[-1])
    prev_macd = get_val(df['MACD'].iloc[-2]) if len(df) > 1 else current_macd
    current_rsi = get_val(df['RSI'].iloc[-1])
    prev_rsi = get_val(df['RSI'].iloc[-2]) if len(df) > 1 else current_rsi
    current_volume = get_val(df['Volume'].iloc[-1])
    vol_5d_avg = get_val(df['Volume'].rolling(window=5).mean().iloc[-1])

    sell_signals = []
    
    if current_price < current_sma_25:
        sell_signals.append("🚨 **【優先度: 高】 25日線を下回りました。**\n中期トレンドの崩壊を示唆します。スイングトレードにおいては、含み損拡大を防ぐための早急な手仕舞い（損切り・微益撤退）を強く推奨します。")
    if current_price < current_open and current_volume > vol_5d_avg:
         sell_signals.append("🚨 **【優先度: 高】 大商いで陰線をつけています。**\n平均より多い出来高を伴って下落しています。大口投資家が売り抜けている可能性があり、急落リスクに警戒が必要です。")
    if current_macd < current_macd_signal or current_macd < prev_macd:
        sell_signals.append("⚠️ **【優先度: 中】 MACDが下落転換しました。**\n上昇の勢い（モメンタム）が低下しています。そろそろ上値が重くなる可能性があるため、利益確定の準備をおすすめします。")
    if current_rsi < prev_rsi and prev_rsi >= 70:
        sell_signals.append("⚠️ **【優先度: 中】 RSIが過熱圏(70超)から反落しました。**\n「買われすぎ」状態から実際に売りが出始めたサインです。短期的な天井を打った可能性があります。")
    if current_price < current_sma_5:
        sell_signals.append("💡 **【優先度: 低】 5日線を下回りました。**\n短期的な上昇トレンドの一服を示します。単独のサインであれば一時的なノイズの可能性もありますが、他の警戒サインと重なる場合は注意が必要です。")

    # --- 売りサイン結果表示 ---
    st.markdown(f"#### 💡 {selected_name} の手仕舞い（売り）サイン診断")
    
    high_alert = any("高" in s for s in sell_signals)
    mid_alert = any("中" in s for s in sell_signals)
    
    if not sell_signals:
        st.success(f"現在、**{selected_name}** に目立った売りサインは点灯していません。トレンドは継続している可能性があります。")
        st.info("💡 **【推奨アクション: ホールド】** 現在のトレンドに乗ったまま、設定した利確・損切ライン（下記）での決済を待ちます。")
    else:
        st.warning(f"現在、**{len(sell_signals)}つ** の売り警戒サインが点灯しています。")
        for signal in sell_signals:
            st.markdown(f"- {signal}")
            
        if high_alert:
            st.error("🚨 **【推奨アクション: 即時撤退】** 中期トレンドの崩壊や大口の売りの兆候があります。**翌営業日の寄り付き（成行）**、または現在値付近での即時決済（損切り・利確）を強く推奨します。")
        elif mid_alert:
            stop_level = current_sma_5
            st.warning(f"⚠️ **【推奨アクション: 逆指値の引き上げ】** 上値が重くなってきたサインです。利益を確保（または損失を最小化）するため、**5日移動平均線（約 ¥{stop_level:,.0f}）** や直近の安値を割ったら決済されるよう、逆指値注文を設定して様子を見ることをお勧めします。")
        else:
            st.info("💡 **【推奨アクション: 警戒しつつホールド】** 短期的な調整の可能性がありますが、致命的なサインではありません。下記の損切りラインを厳守しつつ様子を見ます。")

    # --- 売却目標株価の計算と表示 ---
    base_price = entry_price if entry_price > 0 else current_price
    target_price_sell = base_price * (1 + profit_target_pct / 100)
    stop_price_sell = base_price * (1 + stop_loss_pct / 100)
    
    base_text = f"取得単価 (¥{base_price:,.0f})" if entry_price > 0 else f"現在の株価 (¥{base_price:,.0f})"
    
    st.info(f"**🎯 {base_text} を基準とした売却目安 (ルールベース)**\n"
            f"- 🟢 **利確目標 (+{profit_target_pct}%): ¥{target_price_sell:,.0f}**\n"
            f"- 🔴 **損切ライン ({stop_loss_pct}%): ¥{stop_price_sell:,.0f}**\n\n"
            f"※上記は事前のルールに基づく機械的な目安です。強い売りサインが点灯している場合は、上記の「推奨アクション」を優先してご判断ください。")

    st.markdown("---")
            
    plot_df = df.tail(120) 
    
    fig = go.Figure(data=[go.Candlestick(x=plot_df.index,
                    open=plot_df['Open'].squeeze(),
                    high=plot_df['High'].squeeze(),
                    low=plot_df['Low'].squeeze(),
                    close=plot_df['Close'].squeeze(),
                    name='ローソク足')])
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_5'].squeeze(), line=dict(color='orange', width=1), name='5日移動平均'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_25'].squeeze(), line=dict(color='blue', width=1), name='25日移動平均'))
    
    fig.update_layout(
        title=f"{selected_name} ({selected_stock['ticker']}) の株価推移",
        yaxis_title="株価 (円)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    failure_prob = 100 - selected_stock['score_rule']
    st.info(f"**AIの分析 ({selected_name})**:\n\n"
            f"**【ルールベース判定】** {holding_period}営業日以内に{stop_loss_pct}%の損切りに触れず、+{profit_target_pct}%の利確に到達する確率: **{selected_stock['score_rule']:.1f}%** (失敗する確率: **{failure_prob:.1f}%**)\n\n"
            f"※この確率は、過去3年間の値動きパターン（テクニカル指標）からランダムフォレストが算出した期待値です。")
