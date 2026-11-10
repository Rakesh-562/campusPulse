"""Apriori wrapper: support / confidence / lift, filtered."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_rules(onehot: pd.DataFrame, min_support=0.05, min_lift=1.2, min_conf=0.6, consequent_prefix=None):
    fi = apriori(onehot.astype(bool), min_support=min_support, use_colnames=True)
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[r["lift"] >= min_lift]
    if consequent_prefix:
        r = r[r["consequents"].apply(lambda s: any(str(i).startswith(consequent_prefix) for i in s))]
    r = r.assign(antecedents=r["antecedents"].apply(lambda s: ", ".join(sorted(s))),
                 consequents=r["consequents"].apply(lambda s: ", ".join(sorted(s))))
    return r.sort_values("lift", ascending=False)[["antecedents", "consequents", "support", "confidence", "lift"]]
