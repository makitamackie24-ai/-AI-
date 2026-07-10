import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(
    page_title="日本株AIレコメンドエンジン",
    layout="wide"
)

st.title("日本株 AIトレンド予測＆レコメンドエンジン")
st.write("過去の株価データからテクニカル指標を計算し、機械学習（ランダムフォレスト）を用いて翌日、翌々日（2営業日後）、および1週間後（5営業日後）の株価が上昇する確率と、各価格を予測します。")

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
    df['Return'] = df['Close'].pct_change()
    
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
def analyze_stock_data(df, n_estimators=100):
    if len(df) < 50:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = add_technical_indicators(df)
    
    # 目的変数
    df['Target_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_Class_2D'] = (df['Close'].shift(-2) > df['Close']).astype(int)
    df['Target_Class_1W'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df['Target_Reg'] = df['Close'].shift(-1)
    df['Target_Reg_Open'] = df['Open'].shift(-1)
    df['Target_Reg_High'] = df['High'].shift(-1)
    df['Target_Reg_Low'] = df['Low'].shift(-1)
    df['Target_Reg_Open_2D'] = df['Open'].shift(-2)
    
    df_clean = df.dropna()
    
    features = ['Close', 'Volume', 'SMA_5', 'SMA_25', 'Return', 'RSI', 'Volatility', 'Vol_Change']
    X = df_clean[features]
    
    # モデルの構築と学習 (木の数を変数化)
    model_class = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Class'])
    model_class_2d = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Class_2D'])
    model_class_1w = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Class_1W'])
    model_reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg'])
    model_reg_open = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_Open'])
    model_reg_high = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_High'])
    model_reg_low = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_Low'])
    model_reg_open_2d = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_Open_2D'])
    
    latest_data = df.iloc[-1:][features]
    prev_data = df.iloc[-2:-1][features]
    
    if latest_data.isnull().values.any() or prev_data.isnull().values.any():
         return None
         
    # 予測の実行
    prediction_proba = model_class.predict_proba(latest_data)[0][1] 
    prev_prediction_proba = model_class.predict_proba(prev_data)[0][1]
    prediction_proba_2d = model_class_2d.predict_proba(latest_data)[0][1]
    prev_prediction_proba_2d = model_class_2d.predict_proba(prev_data)[0][1]
    prediction_proba_1w = model_class_1w.predict_proba(latest_data)[0][1]
    prev_prediction_proba_1w = model_class_1w.predict_proba(prev_data)[0][1]
    
    predicted_price = model_reg.predict(latest_data)[0]
    prev_predicted_price = model_reg.predict(prev_data)[0]
    predicted_open = model_reg_open.predict(latest_data)[0]
    prev_predicted_open = model_reg_open.predict(prev_data)[0]
    predicted_high = model_reg_high.predict(latest_data)[0]
    prev_predicted_high = model_reg_high.predict(prev_data)[0]
    predicted_low = model_reg_low.predict(latest_data)[0]
    prev_predicted_low = model_reg_low.predict(prev_data)[0]
    predicted_open_2d = model_reg_open_2d.predict(latest_data)[0]
    prev_predicted_open_2d = model_reg_open_2d.predict(prev_data)[0]
    
    actual_up = float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2])
    
    return {
        "df": df,
        "proba": prediction_proba,
        "prev_proba": prev_prediction_proba,
        "proba_2d": prediction_proba_2d,
        "prev_proba_2d": prev_prediction_proba_2d,
        "proba_1w": prediction_proba_1w,
        "prev_proba_1w": prev_prediction_proba_1w,
        "actual_up": actual_up,
        "pred_close": predicted_price,
        "prev_pred_close": prev_predicted_price,
        "pred_open": predicted_open,
        "prev_pred_open": prev_predicted_open,
        "pred_high": predicted_high,
        "prev_pred_high": prev_predicted_high,
        "pred_low": predicted_low,
        "prev_pred_low": prev_predicted_low,
        "pred_open_2d": predicted_open_2d,
        "prev_pred_open_2d": prev_predicted_open_2d
    }

# --- 全銘柄の分析処理を一括キャッシュ（24時間保持） ---
@st.cache_data(ttl=86400, show_spinner=False)
def generate_all_results(years=3.0, n_estimators=100, top_n=134):
    results = []
    progress_bar = st.progress(0)
    
    tickers = list(TARGET_STOCKS.keys())
    end_date = datetime.today()
    # 取得期間を変数化
    start_date = end_date - timedelta(days=int(365 * years))
    
    # 全銘柄のデータを一括ダウンロード（通信回数を1回に減らし高速化）
    df_all = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True, progress=False)
    
    # --- 売買代金の計算と上位抽出 ---
    trading_values = {}
    valid_tickers = []
    
    for ticker in tickers:
        try:
            # 取得した一括データから対象銘柄のデータを抽出
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
    for i, ticker in enumerate(sorted_tickers):
        name = TARGET_STOCKS[ticker]
        df_ticker = trading_values[ticker]['df']
            
        # 木の数を渡して分析実行
        analysis = analyze_stock_data(df_ticker, n_estimators=n_estimators)
        
        if analysis is not None:
            df = analysis["df"]
            
            # Seriesや配列の対応
            def get_val(val):
                return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

            current_price = get_val(df['Close'].iloc[-1])
            current_open = get_val(df['Open'].iloc[-1])
            current_high = get_val(df['High'].iloc[-1])
            current_low = get_val(df['Low'].iloc[-1])
            
            current_sma_25 = get_val(df['SMA_25'].iloc[-1])
            current_volume = get_val(df['Volume'].iloc[-1])
            vol_change = get_val(df['Vol_Change'].iloc[-1])
            vol_5d_avg = get_val(df['Volume'].rolling(window=5).mean().iloc[-1])
            current_rsi = get_val(df['RSI'].iloc[-1])
            
            # 対象日付の計算
            latest_date_dt = df.index[-1]
            next_date_dt = latest_date_dt + pd.offsets.BDay(1)
            two_days_later_dt = latest_date_dt + pd.offsets.BDay(2)
            prev_date_dt = df.index[-2]
            week_later_dt = latest_date_dt + pd.offsets.BDay(5)
            
            results.append({
                "ticker": ticker,
                "name": name,
                "price": current_price,
                "open": current_open,
                "high": current_high,
                "low": current_low,
                "current_sma_25": current_sma_25,
                "current_volume": current_volume,
                "vol_change": vol_change,
                "vol_5d_avg": vol_5d_avg,
                "current_rsi": current_rsi,
                "latest_date": latest_date_dt.strftime('%Y/%m/%d'),
                "next_date": next_date_dt.strftime('%Y/%m/%d'),
                "two_days_later_date": two_days_later_dt.strftime('%Y/%m/%d'),
                "prev_date": prev_date_dt.strftime('%Y/%m/%d'),
                "week_later_date": week_later_dt.strftime('%Y/%m/%d'),
                "score": analysis["proba"] * 100,
                "prev_score": analysis["prev_proba"] * 100,
                "score_2d": analysis["proba_2d"] * 100,
                "prev_score_2d": analysis["prev_proba_2d"] * 100,
                "score_1w": analysis["proba_1w"] * 100,
                "prev_score_1w": analysis["prev_proba_1w"] * 100,
                "actual_up": analysis["actual_up"],
                "pred_close": analysis["pred_close"],
                "prev_pred_close": analysis["prev_pred_close"], 
                "pred_open": analysis["pred_open"],
                "prev_pred_open": analysis["prev_pred_open"], 
                "pred_high": analysis["pred_high"],
                "prev_pred_high": analysis["prev_pred_high"],
                "pred_low": analysis["pred_low"],
                "prev_pred_low": analysis["prev_pred_low"],
                "pred_open_2d": analysis["pred_open_2d"],
                "prev_pred_open_2d": analysis["prev_pred_open_2d"],
                "pred_diff": analysis["pred_high"] - analysis["pred_low"],
                "pred_diff_pct": (analysis["pred_high"] - analysis["pred_low"]) / current_price * 100 if current_price > 0 else 0,
                "pred_open_2d_pct": (analysis["pred_open_2d"] - analysis["pred_open"]) / analysis["pred_open"] * 100 if analysis["pred_open"] > 0 else 0,
                "df": df
            })
        progress_bar.progress((i + 1) / len(sorted_tickers))
        
    progress_bar.empty()
    return results

# --- メイン処理 ---
st.markdown("### 分析設定")
top_n = st.slider(f"解析対象とする売買代金上位の企業数を選択 (最大{stock_count}社)", min_value=10, max_value=stock_count, value=150, step=10, help="全銘柄から直近の売買代金が多い企業を自動選出し、AI解析の対象を絞ります。数を減らすと計算時間が短縮されます。")

col1, col2 = st.columns([1, 2])

with col1:
    run_btn = st.button("AI分析を実行 / 結果を表示", type="primary", help="高精度な詳細分析（約3年分のデータ・木100本）を行います。")
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
    n_estimators = 100
    
    with st.spinner(f"対象全社のデータを取得し、売買代金上位{top_n}社をAIモデルで詳細分析中...\n（計算済みの場合は一瞬で表示されます）"):
        results = generate_all_results(years=years, n_estimators=n_estimators, top_n=top_n)
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        st.session_state['analysis_results'] = results_sorted
        st.success(f"売買代金上位 {len(results)}社の詳細分析が完了（またはキャッシュから取得）しました！")

# --- 結果の表示 ---
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    # --- 厳選AIレコメンドセクション ---
    st.subheader("厳選AIレコメンド銘柄")
    st.write("設定された厳しい条件（中期・短期の上昇期待、翌日予測の方向性一致、高い予測精度、十分なボラティリティ）をすべてクリアした有望銘柄をピックアップします。")
    st.caption("【選定条件】・1週間後＆翌々日の上昇確率50%超 ・翌日の上昇確率と予測終値の方向が一致 ・AI予測誤差3%未満 ・翌日予測変動幅1.5%以上")
    
    recommended_stocks = []
    for stock in results:
        # 条件1: 1週間後上昇確率が高い (50%超)
        cond1 = stock['score_1w'] > 50
        # 条件2: 翌々日の上昇確率が高い (50%超)
        cond2 = stock['score_2d'] > 50
        # 条件3: 「翌日の上昇確率が高いかつ予測終値が現在より高い」または「翌日の上昇確率が低いかつ予測終値が現在より低い」
        cond3_up = (stock['score'] > 50) and (stock['pred_close'] > stock['price'])
        cond3_down = (stock['score'] <= 50) and (stock['pred_close'] <= stock['price'])
        cond3 = cond3_up or cond3_down
        # 条件4: 直近の予測の答え合わせの誤差が少ない (誤差率3%未満)
        error_rate = abs(stock['price'] - stock['prev_pred_close']) / stock['price'] * 100 if stock['price'] > 0 else 100
        cond4 = error_rate < 3.0
        # 条件5: 翌日変動幅予測が大きい (1.5%以上)
        cond5 = stock['pred_diff_pct'] >= 1.5
        
        if cond1 and cond2 and cond3 and cond4 and cond5:
            # 総合スコアの計算（ソート用）
            total_score = stock['score_1w'] + stock['score_2d'] + (stock['pred_diff_pct'] * 5) - (error_rate * 5)
            stock['recommend_score'] = total_score
            stock['error_rate'] = error_rate
            recommended_stocks.append(stock)
            
    # スコア順にソートし、最大5社取得
    recommended_stocks = sorted(recommended_stocks, key=lambda x: x['recommend_score'], reverse=True)[:5]
    
    if len(recommended_stocks) > 0:
        rec_cols = st.columns(len(recommended_stocks))
        for i, stock in enumerate(recommended_stocks):
            with rec_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{stock['name']}**")
                    st.caption(f"{stock['ticker']}")
                    
                    # RSIの表示と過熱感アラート
                    if stock['current_rsi'] >= 70:
                        rsi_color = "red"
                        rsi_alert = " 過熱感注意"
                    else:
                        rsi_color = "gray"
                        rsi_alert = ""
                    st.markdown(f"<p style='color: {rsi_color}; font-size: 0.85em; font-weight: bold; margin-bottom: 5px;'>RSI: {stock['current_rsi']:.1f}%{rsi_alert}</p>", unsafe_allow_html=True)

                    # 移動平均線と出来高のアラート
                    alerts = []
                    if stock['price'] < stock['current_sma_25']:
                        alerts.append("25日線割れ")
                    if stock['current_volume'] < stock['vol_5d_avg']:
                        alerts.append("出来高低迷")
                    if alerts:
                        st.markdown(f"<p style='color: red; font-size: 0.85em; font-weight: bold; margin-top: -5px; margin-bottom: 5px;'>注意: {' / '.join(alerts)}</p>", unsafe_allow_html=True)

                    st.markdown(f"<p style='color: green; font-weight: bold; margin-bottom: 0px;'>1週間後: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: green; font-weight: bold; margin-bottom: 0px;'>翌々日: {stock['score_2d']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: orange; font-weight: bold; margin-bottom: 0px;'>変動幅: {stock['pred_diff_pct']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: gray; font-size: 0.8em; margin-bottom: 0px;'>予測誤差: {stock['error_rate']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px;'><b>直近 ({stock['latest_date']})</b><br>終値: ¥{stock['price']:,.0f} | 始値: ¥{stock['open']:,.0f}<br>高値: ¥{stock['high']:,.0f} | 安値: ¥{stock['low']:,.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px; margin-top: 5px;'><b>予測 ({stock['next_date']})</b><br>終値: ¥{stock['pred_close']:,.0f} | 始値: ¥{stock['pred_open']:,.0f}<br>高値: ¥{stock['pred_high']:,.0f} | 安値: ¥{stock['pred_low']:,.0f}</p>", unsafe_allow_html=True)
    else:
        st.info("現在、すべての厳選条件を満たす銘柄はありません。相場環境が変わるのをお待ちください。")
        
    st.divider()

    # --- 厳選AIレコメンド銘柄（プチ株用）セクション ---
    st.subheader("プチ株用 厳選AIレコメンド銘柄")
    st.write("翌日の始値で買い、翌々日の始値で売るような短期トレード（プチ株など）に向いた銘柄をピックアップします。")
    st.caption("【選定条件】・翌日始値→翌々日始値の上昇率がプラス ・1週間後＆翌々日の上昇確率50%超 ・翌日の上昇確率と予測終値の方向が一致 ・AI予測誤差3%未満")
    
    recommended_stocks_petit = []
    for stock in results:
        # 条件1: 1週間後上昇確率が高い (50%超)
        cond1 = stock['score_1w'] > 50
        # 条件2: 翌々日の上昇確率が高い (50%超)
        cond2 = stock['score_2d'] > 50
        # 条件3: 「翌日の上昇確率が高いかつ予測終値が現在より高い」または「翌日の上昇確率が低いかつ予測終値が現在より低い」
        cond3_up = (stock['score'] > 50) and (stock['pred_close'] > stock['price'])
        cond3_down = (stock['score'] <= 50) and (stock['pred_close'] <= stock['price'])
        cond3 = cond3_up or cond3_down
        # 条件4: 直近の予測の答え合わせの誤差が少ない (誤差率3%未満)
        error_rate = abs(stock['price'] - stock['prev_pred_close']) / stock['price'] * 100 if stock['price'] > 0 else 100
        cond4 = error_rate < 3.0
        # 条件5: 翌日始値から翌々日の始値の上昇率が高い (プラスであること)
        cond5_petit = stock['pred_open_2d_pct'] > 0
        
        if cond1 and cond2 and cond3 and cond4 and cond5_petit:
            stock['recommend_score_petit'] = stock['pred_open_2d_pct']
            stock['error_rate'] = error_rate
            recommended_stocks_petit.append(stock)
            
    # 上昇率順にソートし、最大5社取得
    recommended_stocks_petit = sorted(recommended_stocks_petit, key=lambda x: x['recommend_score_petit'], reverse=True)[:5]
    
    if len(recommended_stocks_petit) > 0:
        rec_cols_petit = st.columns(len(recommended_stocks_petit))
        for i, stock in enumerate(recommended_stocks_petit):
            with rec_cols_petit[i]:
                with st.container(border=True):
                    st.markdown(f"**{stock['name']}**")
                    st.caption(f"{stock['ticker']}")
                    
                    # RSIの表示と過熱感アラート
                    if stock['current_rsi'] >= 70:
                        rsi_color = "red"
                        rsi_alert = " 過熱感注意"
                    else:
                        rsi_color = "gray"
                        rsi_alert = ""
                    st.markdown(f"<p style='color: {rsi_color}; font-size: 0.85em; font-weight: bold; margin-bottom: 5px;'>RSI: {stock['current_rsi']:.1f}%{rsi_alert}</p>", unsafe_allow_html=True)

                    st.markdown(f"<p style='color: #e91e63; font-weight: bold; margin-bottom: 0px;'>始値上昇率: {stock['pred_open_2d_pct']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: green; font-size: 0.8em; margin-bottom: 0px;'>1週間後上昇確率: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: gray; font-size: 0.8em; margin-bottom: 0px;'>予測誤差: {stock['error_rate']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px;'><b>予測 ({stock['next_date']})</b><br>始値: ¥{stock['pred_open']:,.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px; margin-top: 5px;'><b>予測 ({stock['two_days_later_date']})</b><br>始値: ¥{stock['pred_open_2d']:,.0f}</p>", unsafe_allow_html=True)
    else:
        st.info("現在、プチ株用の厳選条件を満たす銘柄はありません。")

    st.divider()

    # --- ランキング切り替えUI ---
    sort_basis = st.radio(
        "ランキングの基準を選択:",
        ("翌日予測ベース", "翌々日予測ベース", "1週間後予測ベース", "翌日変動幅(%)予測ベース", "翌日始値→翌々日始値上昇率ベース"),
        horizontal=True
    )
    
    if sort_basis == "翌日予測ベース":
        sort_key = 'score'
    elif sort_basis == "翌々日予測ベース":
        sort_key = 'score_2d'
    elif sort_basis == "1週間後予測ベース":
        sort_key = 'score_1w'
    elif sort_basis == "翌日変動幅(%)予測ベース":
        sort_key = 'pred_diff_pct'
    else:
        sort_key = 'pred_open_2d_pct'
        
    results_sorted = sorted(results, key=lambda x: x[sort_key], reverse=True)
    
    col_up, col_down = st.columns(2)
    
    with col_up:
        if sort_basis == "翌日変動幅(%)予測ベース":
            st.subheader(f"価格変動幅(大) トップ10")
        elif sort_basis == "翌日始値→翌々日始値上昇率ベース":
            st.subheader(f"翌日始値→翌々日始値上昇率 トップ10")
        else:
            st.subheader(f"上昇期待度 トップ10 ({sort_basis})")
            
        for i in range(min(10, len(results_sorted))):
            stock = results_sorted[i]
            with st.container(border=True):
                st.markdown(f"**第{i+1}位: {stock['name']}**")
                st.caption(f"{stock['ticker']} | {stock['latest_date']} 終値: ¥{stock['price']:,.0f} (始値: ¥{stock['open']:,.0f} / 高値: ¥{stock['high']:,.0f} / 安値: ¥{stock['low']:,.0f})")
                
                if sort_basis == "翌日予測ベース":
                    st.markdown(f"<h3 style='color: green; margin-top: -10px; margin-bottom: 0px;'>翌日上昇確率: {stock['score']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold; margin-bottom: 0px;'>翌々日({stock['two_days_later_date']})上昇確率: {stock['score_2d']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>1週間後({stock['week_later_date']})上昇確率: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "翌々日予測ベース":
                    st.markdown(f"<h3 style='color: green; margin-top: -10px; margin-bottom: 0px;'>翌々日({stock['two_days_later_date']})上昇確率: {stock['score_2d']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold; margin-bottom: 0px;'>翌日上昇確率: {stock['score']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>1週間後({stock['week_later_date']})上昇確率: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "1週間後予測ベース":
                    st.markdown(f"<h3 style='color: green; margin-top: -10px; margin-bottom: 0px;'>1週間後({stock['week_later_date']})上昇確率: {stock['score_1w']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold; margin-bottom: 0px;'>翌日上昇確率: {stock['score']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>翌々日({stock['two_days_later_date']})上昇確率: {stock['score_2d']:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "翌日変動幅(%)予測ベース":
                    st.markdown(f"<h3 style='color: orange; margin-top: -10px; margin-bottom: 0px;'>予測変動幅: {stock['pred_diff_pct']:.1f}% (¥{stock['pred_diff']:,.0f})</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold; margin-bottom: 0px;'>翌日上昇確率: {stock['score']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>翌々日({stock['two_days_later_date']})上昇確率: {stock['score_2d']:.1f}%</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: #e91e63; margin-top: -10px; margin-bottom: 0px;'>始値上昇率: {stock['pred_open_2d_pct']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold; margin-bottom: 0px;'>翌日始値予測: ¥{stock['pred_open']:,.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #2e7d32; font-weight: bold;'>翌々日始値予測: ¥{stock['pred_open_2d']:,.0f}</p>", unsafe_allow_html=True)
                    
                st.markdown(f"**{stock['next_date']} 予測**<br>終値: **¥{stock['pred_close']:,.0f}** | 始値: **¥{stock['pred_open']:,.0f}** | 高値: **¥{stock['pred_high']:,.0f}** | 安値: **¥{stock['pred_low']:,.0f}**", unsafe_allow_html=True)

    with col_down:
        if sort_basis == "翌日変動幅(%)予測ベース":
            st.subheader(f"価格変動幅(小) ワースト10")
        elif sort_basis == "翌日始値→翌々日始値上昇率ベース":
            st.subheader(f"翌日始値→翌々日始値上昇率 ワースト10")
        else:
            st.subheader(f"下落警戒 ワースト10 ({sort_basis})")
            
        for i in range(min(10, len(results_sorted))):
            stock = results_sorted[-(i+1)]
            down_prob = 100 - stock['score']
            down_prob_2d = 100 - stock['score_2d']
            down_prob_1w = 100 - stock['score_1w']
            
            with st.container(border=True):
                st.markdown(f"**第{i+1}位: {stock['name']}**")
                st.caption(f"{stock['ticker']} | {stock['latest_date']} 終値: ¥{stock['price']:,.0f} (始値: ¥{stock['open']:,.0f} / 高値: ¥{stock['high']:,.0f} / 安値: ¥{stock['low']:,.0f})")
                
                if sort_basis == "翌日予測ベース":
                    st.markdown(f"<h3 style='color: red; margin-top: -10px; margin-bottom: 0px;'>翌日下落確率: {down_prob:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold; margin-bottom: 0px;'>翌々日({stock['two_days_later_date']})下落確率: {down_prob_2d:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>1週間後({stock['week_later_date']})下落確率: {down_prob_1w:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "翌々日予測ベース":
                    st.markdown(f"<h3 style='color: red; margin-top: -10px; margin-bottom: 0px;'>翌々日({stock['two_days_later_date']})下落確率: {down_prob_2d:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold; margin-bottom: 0px;'>翌日下落確率: {down_prob:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>1週間後({stock['week_later_date']})下落確率: {down_prob_1w:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "1週間後予測ベース":
                    st.markdown(f"<h3 style='color: red; margin-top: -10px; margin-bottom: 0px;'>1週間後({stock['week_later_date']})下落確率: {down_prob_1w:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold; margin-bottom: 0px;'>翌日下落確率: {down_prob:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>翌々日({stock['two_days_later_date']})下落確率: {down_prob_2d:.1f}%</p>", unsafe_allow_html=True)
                elif sort_basis == "翌日変動幅(%)予測ベース":
                    st.markdown(f"<h3 style='color: orange; margin-top: -10px; margin-bottom: 0px;'>予測変動幅: {stock['pred_diff_pct']:.1f}% (¥{stock['pred_diff']:,.0f})</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold; margin-bottom: 0px;'>翌日下落確率: {down_prob:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>翌々日({stock['two_days_later_date']})下落確率: {down_prob_2d:.1f}%</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: blue; margin-top: -10px; margin-bottom: 0px;'>始値上昇率: {stock['pred_open_2d_pct']:.1f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold; margin-bottom: 0px;'>翌日始値予測: ¥{stock['pred_open']:,.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #c62828; font-weight: bold;'>翌々日始値予測: ¥{stock['pred_open_2d']:,.0f}</p>", unsafe_allow_html=True)
                    
                st.markdown(f"**{stock['next_date']} 予測**<br>終値: **¥{stock['pred_close']:,.0f}** | 始値: **¥{stock['pred_open']:,.0f}** | 高値: **¥{stock['pred_high']:,.0f}** | 安値: **¥{stock['pred_low']:,.0f}**", unsafe_allow_html=True)

    st.divider()
    
    # 個別チャートの表示セクション
    st.subheader("銘柄ごとの詳細チャート確認")
    selected_name = st.selectbox("詳細を見たい銘柄を選択してください", [r['name'] for r in sorted(results, key=lambda x: x['ticker'])])
    selected_stock = next(r for r in results if r['name'] == selected_name)
    df = selected_stock['df']
    
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
    
    down_score = 100 - selected_stock['score']
    down_score_2d = 100 - selected_stock['score_2d']
    down_score_1w = 100 - selected_stock['score_1w']
    st.info(f"**AIの分析 ({selected_name})**:\n\n"
            f"**【翌日 ({selected_stock['next_date']}) のトレンド】** 上がる確率: **{selected_stock['score']:.1f}%** / 下がる確率: **{down_score:.1f}%**\n\n"
            f"**【翌々日 ({selected_stock['two_days_later_date']}) のトレンド】** 上がる確率: **{selected_stock['score_2d']:.1f}%** / 下がる確率: **{down_score_2d:.1f}%**\n\n"
            f"**【1週間後 ({selected_stock['week_later_date']}) のトレンド】** 上がる確率: **{selected_stock['score_1w']:.1f}%** / 下がる確率: **{down_score_1w:.1f}%**\n\n"
            f"**{selected_stock['next_date']} のAI予測価格:**\n"
            f"- 終値: **¥{selected_stock['pred_close']:,.0f}** (現在 ¥{selected_stock['price']:,.0f})\n"
            f"- 始値: **¥{selected_stock['pred_open']:,.0f}** (現在 ¥{selected_stock['open']:,.0f})\n"
            f"- 高値: **¥{selected_stock['pred_high']:,.0f}** (現在 ¥{selected_stock['high']:,.0f})\n"
            f"- 安値: **¥{selected_stock['pred_low']:,.0f}** (現在 ¥{selected_stock['low']:,.0f})\n\n"
            f"**{selected_stock['two_days_later_date']} のAI予測価格:**\n"
            f"- 始値: **¥{selected_stock['pred_open_2d']:,.0f}** (翌日始値からの上昇率: {selected_stock['pred_open_2d_pct']:.1f}%)")
    
    # --- 直近の予測の答え合わせ表示 ---
    st.markdown("### 直近の予測の答え合わせ（翌日予測）")
    st.write(f"1つ前の営業日（{selected_stock['prev_date']}）のデータを使って、最新営業日（{selected_stock['latest_date']}）の株価変動を予測した結果です。")
    
    if 'prev_score' in selected_stock:
        prev_score = selected_stock['prev_score']
        actual_result_text = "上昇" if selected_stock['actual_up'] else "下落/変わらず"
        prediction_text = "上昇" if prev_score > 50 else "下落"
        is_correct = (prev_score > 50) == selected_stock['actual_up']
        
        if is_correct:
            st.success(f"**方向的中！** \n\n予測方向: **{prediction_text}** (確率 {prev_score:.1f}%) ➔ 実際: **{actual_result_text}**")
        else:
            st.error(f"**方向ハズレ** \n\n予測方向: **{prediction_text}** (確率 {prev_score:.1f}%) ➔ 実際: **{actual_result_text}**")
            
        st.write(f"**価格の答え合わせ ({selected_stock['latest_date']}):**")
        st.write(f"- 【終値】AI予測: **¥{selected_stock['prev_pred_close']:,.0f}** ➔ 実際: **¥{selected_stock['price']:,.0f}** (誤差: ¥{abs(selected_stock['price'] - selected_stock['prev_pred_close']):,.0f})")
        st.write(f"- 【始値】AI予測: **¥{selected_stock['prev_pred_open']:,.0f}** ➔ 実際: **¥{selected_stock['open']:,.0f}** (誤差: ¥{abs(selected_stock['open'] - selected_stock['prev_pred_open']):,.0f})")
        st.write(f"- 【高値】AI予測: **¥{selected_stock['prev_pred_high']:,.0f}** ➔ 実際: **¥{selected_stock['high']:,.0f}** (誤差: ¥{abs(selected_stock['high'] - selected_stock['prev_pred_high']):,.0f})")
        st.write(f"- 【安値】AI予測: **¥{selected_stock['prev_pred_low']:,.0f}** ➔ 実際: **¥{selected_stock['low']:,.0f}** (誤差: ¥{abs(selected_stock['low'] - selected_stock['prev_pred_low']):,.0f})")
    else:
        st.warning("新しい予測データが読み込まれていません。再度「分析を開始する」ボタンを押してデータを更新してください。")
