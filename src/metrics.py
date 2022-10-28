import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


def confusion_matrix_plot(true, pred, columns) -> None:
    conf_matrix = confusion_matrix(
        np.argmax(true, axis=1),
        np.argmax(pred, axis=1)
    )

    df_cm = pd.DataFrame(
        conf_matrix,
        columns=columns,
        index=columns
    )
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    sns.heatmap(
        df_cm,
        cbar=False,
        annot=True,
        cmap=cmap,
        square=True,
        fmt='.0f',
        annot_kws={'size': 10}
    )


def calculate_roc_auc(true: np.ndarray, pred: np.ndarray) -> None:
    macro_roc_auc_ovo = roc_auc_score(true, pred, multi_class="ovo", average="macro")
    micro_roc_auc_ovo = roc_auc_score(true, pred, multi_class="ovo", average="micro")
    print(
        f"""
    roc auc macro: {round(macro_roc_auc_ovo, 4)},
    roc auc micro: {round(micro_roc_auc_ovo, 4)}
    """
    )


def calculate_recall(true: np.ndarray, pred: np.ndarray) -> None:
    recall = recall_score(true, pred, average="macro")
    print(
        f"""
    recall: {round(recall, 4)}
    """
    )


def calculate_precision(true: np.ndarray, pred: np.ndarray) -> None:
    precision = precision_score(true, pred, average="macro")
    print(
        f"""
    precision: {round(precision, 4)}
    """
    )
