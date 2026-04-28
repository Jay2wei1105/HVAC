# engine/core/ml/analysis.py
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def correlation_report(feat_df, target, top_k=20):
    """訓練前快速掃描：Pearson / Spearman / MI 三合一。"""
    # 排除目標變數為空或無限大的列
    valid_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    
    num = valid_df.select_dtypes("number").drop(columns=[target])
    y = valid_df[target]
    
    # 再次檢查是否有數據，防止全部被 drop 掉
    if len(valid_df) == 0:
        return pd.DataFrame()
        
    pear  = num.corrwith(y).abs().rename("pearson_abs")
    spear = num.corrwith(y, method="spearman").abs().rename("spearman_abs")
    mi    = pd.Series(
        mutual_info_regression(num.fillna(0), y, random_state=42),
        index=num.columns, name="mutual_info")
    
    report = pd.concat([pear, spear, mi], axis=1)
    report["abs_correlation"] = report["pearson_abs"] # UI 用
    report["rank_score"] = report.rank().sum(axis=1)
    return report.sort_values("rank_score", ascending=False).head(top_k)

def suspicious_features(report, pearson_thr=0.02, mi_thr=0.01):
    mask = ((report["pearson_abs"] < pearson_thr)
            & (report["spearman_abs"] < pearson_thr)
            & (report["mutual_info"] < mi_thr))
    return report[mask].index.tolist()
