
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


######################################################
# Görev 1: Veriyi Hazırlama
#######################################################

df_ = pd.read_excel("recommender_systems/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df.head()
df.isnull().sum()


df = df[df["StockCode"] != "POST"]
df.shape
df.describe().T


df.dropna(inplace=True)


df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T



df = df[df["Price"] > 0]
df.describe().T


df.describe([0.5, 0.75, 0.95, 0.99]).T
# sns.boxplot() ile bunu yapabiliriz
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe([0.5, 0.75, 0.95, 0.99]).T

###################################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
####################################################################

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


rules = create_rules(df, country="Germany")

#################################################################################
# Görev 3: Sepet İçerisindeki Ürün Id’leriVerilen Kullanıcılara Ürün Önerisinde Bulunma
##################################################################################

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


# Kullanıcı1’in sepetindebulunanürününid'si: 21987
# Kullanıcı2’in sepetindebulunanürününid'si: 23235
# Kullanıcı3’in sepetindebulunanürününid'si: 22747

check_id(df, 21987)
check_id(df, 23235)
check_id(df, 22747)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


a = arl_recommender(rules, 21987, 1)
check_id(df, int(a[0]))
b = arl_recommender(rules, 23235, 1)
check_id(df, int(b[0]))
c = arl_recommender(rules, 22747, 2)
check_id(df, int(c[1]))