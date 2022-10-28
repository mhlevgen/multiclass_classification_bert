import re
import string
from typing import Optional

import pandas as pd


def remove_punctuation(text: str) -> Optional[str]:
    if not pd.isna(text):
        text = re.sub(f"[{string.punctuation}]+", ' ', text).lower()
        return text.strip()
    return None


def concat_texts(*, main_text: str, add_text: str, manufacturer: str) -> str:
    return f"{manufacturer} {main_text} {add_text}"


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['add_text_processed'] = data['add_text'].apply(remove_punctuation)
    data['main_text_processed'] = data['main_text'].apply(remove_punctuation)
    data['manufacturer_processed'] = data['manufacturer'].apply(remove_punctuation)

    data['concat_text'] = data.apply(
        lambda x: concat_texts(
            main_text=x['main_text_processed'],
            add_text=x['add_text_processed'],
            manufacturer=x['manufacturer_processed']
        ),
        axis=1)

    return data
