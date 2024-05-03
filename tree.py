import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import networkx as nx
import csv

# Nhập tên file CSV
file = str(input("Nhập tên file: "))

# Đọc dữ liệu từ file CSV
with open(file, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    output = []
    for row in csvreader:
        output.append(row)

# Tạo DataFrame
df = pd.DataFrame(output)

# Xử lý giá trị None
df = df.applymap(lambda x: '' if x is None else x)

# Mã hóa dữ liệu
encoded_df = pd.get_dummies(df.apply(lambda x: ','.join(x), axis=1).str.get_dummies(','))

# Nhập giá trị min_support
minsupportInput = float(input("Nhập min support: "))

# Tìm tập phổ biến
frequent_itemsets = fpgrowth(encoded_df, min_support=minsupportInput, use_colnames=True)

# Thêm cột 'length' để lưu độ dài của mỗi itemset
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Khởi tạo đồ thị
G = nx.Graph()

# Thêm node gốc
G.add_node("Root")

# Kết nối node gốc với tập phổ biến có độ dài 1
for index, row in frequent_itemsets[frequent_itemsets['length'] == 1].iterrows():
    G.add_edge("Root", row['itemsets'], weight=row['support'])

# Thêm các cạnh và trọng số cho tập phổ biến có độ dài lớn hơn 1
for index, row in frequent_itemsets[frequent_itemsets['length'] > 1].iterrows():
    items = list(row['itemsets'])
    for i in range(len(items) - 1):
        item1 = items[i]
        item2 = items[i+1]
        if G.has_edge(item1, item2):
            G[item1][item2]['weight'] += row['support']
        else:
            G.add_edge(item1, item2, weight=row['support'])

# Vẽ đồ thị
pos = nx.spring_layout(G)  # Sử dụng spring layout
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=20)

# Hiển thị trọng số trên các cạnh
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Hiển thị đồ thị
plt.show()