import pandas as pd
from src.config.paths import WEATHER_RAW_DIR
import glob, os

paths = glob.glob(str(WEATHER_RAW_DIR / "*.csv"))
print(paths)

df = pd.read_csv(paths[0], encoding="cp949")   # 첫 파일
print(df.head(20))
col = "일강수량(mm)"  # 실제 컬럼명

print(df[col].unique()[:30])  # 어떤 문자열이 있는지
print(df[col].isna().mean())  # 읽는 순간 NaN 비율
print(df[df[col].astype(str).str.strip() == ""].shape)  # 빈값 몇 개인지
