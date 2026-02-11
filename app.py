import streamlit as st
import pandas as pd
import joblib
from rules import strong_rules

from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("model.json")

feature_columns = joblib.load("feature_columns.pkl")
data = pd.read_csv("data_sample.csv")
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("model.json")


st.title("🛡 保险反欺诈决策引擎 MVP")

report_id = st.text_input("请输入报案号：")

if st.button("评估"):

    row = data[data["report_id"] == report_id]

    if row.empty:
        st.error("未找到该报案号")
    else:
        row = row.iloc[0]

        rule_hits = strong_rules(row)

        score = model.predict_proba(
            row[feature_columns].values.reshape(1, -1)
        )[0][1]

        if rule_hits:
            decision = "建议提调（强规则触发）"
        elif score > 0.7:
            decision = "建议提调（模型高风险）"
        else:
            decision = "无需提调"

        st.metric("风险评分", round(score, 3))
        st.write("决策建议：", decision)

        if rule_hits:
            st.write("命中规则：")
            for r in rule_hits:
                st.write("-", r)