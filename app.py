# ... existing code ...
def analyze_stock_data(df, n_estimators=100):
# ... existing code ...
    df['Target_Reg_Open'] = df['Open'].shift(-1)
    df['Target_Reg_High'] = df['High'].shift(-1)
    df['Target_Reg_Low'] = df['Low'].shift(-1)
    df['Target_Reg_1W'] = df['Close'].shift(-5)
    
    df_clean = df.dropna()
# ... existing code ...
    model_reg_open = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_Open'])
    model_reg_high = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_High'])
    model_reg_low = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_Low'])
    model_reg_1w = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, df_clean['Target_Reg_1W'])
    
    latest_data = df.iloc[-1:][features]
# ... existing code ...
    predicted_high = model_reg_high.predict(latest_data)[0]
    prev_predicted_high = model_reg_high.predict(prev_data)[0]
    predicted_low = model_reg_low.predict(latest_data)[0]
    prev_predicted_low = model_reg_low.predict(prev_data)[0]
    predicted_close_1w = model_reg_1w.predict(latest_data)[0]
    prev_predicted_close_1w = model_reg_1w.predict(prev_data)[0]
    
    actual_up = float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2])
# ... existing code ...
        "pred_high": predicted_high,
        "prev_pred_high": prev_predicted_high,
        "pred_low": predicted_low,
        "prev_pred_low": prev_predicted_low,
        "pred_close_1w": predicted_close_1w,
        "prev_pred_close_1w": prev_predicted_close_1w
    }

# --- 全銘柄の分析処理を一括キャッシュ（24時間保持） ---
# ... existing code ...
                "pred_high": analysis["pred_high"],
# --- 結果の表示 ---
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    # --- 厳選AIレコメンドセクション ---
    st.subheader("厳選AIレコメンド銘柄")
    
    # 予算フィルタを追加 (再計算なしでフィルタ可能にするためここに配置)
    max_price = st.number_input("予算上限：1株あたりの価格（円）を設定してください", min_value=100, max_value=150000, value=5000, step=100, help="指定した金額以下の銘柄のみをレコメンドします。（例: 5000円 = 100株単位で50万円）")
    
    st.write("設定された厳しい条件（中期の上昇期待、高い予測精度、ご指定の予算内）をすべてクリアした有望銘柄をピックアップします。")
    st.caption(f"【選定条件】・1株{max_price:,}円以下 ・1週間後の予測終値が現在値より10%以上高い ・1週間後の上昇確率50%超 ・AI予測誤差3%未満")
    
    recommended_stocks = []
    for stock in results:
        # 条件0: 株価が設定した上限金額以下であること
        cond0 = stock['price'] <= max_price
        # 条件1: 1週間後上昇確率が高い (50%超)
        cond1 = stock['score_1w'] > 50
        # 条件2: 1週間後の予測終値が現在値より10%以上高い
        cond2 = stock['pred_close_1w_pct'] >= 10.0
        # 条件3: 直近の予測の答え合わせの誤差が少ない (誤差率3%未満)
        error_rate = abs(stock['price'] - stock['prev_pred_close']) / stock['price'] * 100 if stock['price'] > 0 else 100
        cond3 = error_rate < 3.0
        
        if cond0 and cond1 and cond2 and cond3:
            # 総合スコアの計算（ソート用）
            total_score = stock['pred_close_1w_pct'] + stock['score_1w'] - error_rate
            stock['recommend_score'] = total_score
            stock['error_rate'] = error_rate
            recommended_stocks.append(stock)
            
    # スコア順にソートし、最大5社取得
# ... existing code ...
                    if alerts:
                        st.markdown(f"<p style='color: red; font-size: 0.85em; font-weight: bold; margin-top: -5px; margin-bottom: 5px;'>注意: {' / '.join(alerts)}</p>", unsafe_allow_html=True)

                    st.markdown(f"<p style='color: green; font-weight: bold; margin-bottom: 0px;'>1週間後上昇確率: {stock['score_1w']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: orange; font-weight: bold; margin-bottom: 0px;'>1週間後予測上昇率: {stock['pred_close_1w_pct']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: gray; font-size: 0.8em; margin-bottom: 0px;'>予測誤差: {stock['error_rate']:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px;'><b>直近 ({stock['latest_date']})</b><br>終値: ¥{stock['price']:,.0f} | 始値: ¥{stock['open']:,.0f}<br>高値: ¥{stock['high']:,.0f} | 安値: ¥{stock['low']:,.0f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.85em; margin-bottom: 0px; margin-top: 5px;'><b>予測 ({stock['week_later_date']})</b><br>終値: ¥{stock['pred_close_1w']:,.0f}</p>", unsafe_allow_html=True)
    else:
        st.info("現在、すべての厳選条件を満たす銘柄はありません。相場環境が変わるのをお待ちください。")
# ... existing code ...
            f"**【1週間後 ({selected_stock['week_later_date']}) のトレンド】** 上がる確率: **{selected_stock['score_1w']:.1f}%** / 下がる確率: **{down_score_1w:.1f}%**\n\n"
            f"**{selected_stock['next_date']} のAI予測価格:**\n"
            f"- 終値: **¥{selected_stock['pred_close']:,.0f}** (現在 ¥{selected_stock['price']:,.0f})\n"
            f"- 始値: **¥{selected_stock['pred_open']:,.0f}** (現在 ¥{selected_stock['open']:,.0f})\n"
            f"- 高値: **¥{selected_stock['pred_high']:,.0f}** (現在 ¥{selected_stock['high']:,.0f})\n"
            f"- 安値: **¥{selected_stock['pred_low']:,.0f}** (現在 ¥{selected_stock['low']:,.0f})\n\n"
            f"**{selected_stock['week_later_date']} のAI予測価格:**\n"
            f"- 終値: **¥{selected_stock['pred_close_1w']:,.0f}** (現在価格からの上昇率: {selected_stock['pred_close_1w_pct']:.1f}%)")
    
    # --- 直近の予測の答え合わせ表示 ---
# ... existing code ...
