import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import networkx as nx
import csv

# Open the CSV file
file = str(input("nhap ten file: "))
with open(file, newline='') as csvfile:
    # Create a CSV reader object with comma as the delimiter
    csvreader = csv.reader(csvfile, delimiter=',')
    # Initialize an empty list to store the rows
    output = []
    # Iterate over each row in the CSV file
    for row in csvreader:
        # Append the row to the output list
        output.append(row)

# Print the output

# Tạo DataFrame từ dữ liệu gốc
df = pd.DataFrame(output)

# Thay thế None bằng chuỗi trống
df = df.applymap(lambda x: '' if x is None else x)

# Mã hoá dữ liệu
encoded_df = pd.get_dummies(df.apply(lambda x: ','.join(x), axis=1).str.get_dummies(','))

minsupportInput =float(input("nhap min support: "))
# Tìm tập phổ biến
frequent_itemsets = fpgrowth(encoded_df, min_support=minsupportInput, use_colnames=True)

# Vẽ cây với trọng số là tần suất xuất hiện
G = nx.Graph()
for index, row in frequent_itemsets.iterrows():
    items = list(row['itemsets'])  # Chuyển đổi thành danh sách
    for i in range(len(items) - 1):
        item1 = items[i]
        item2 = items[i+1]
        if G.has_edge(item1, item2):
            # Tăng trọng số nếu cạnh đã tồn tại
            G[item1][item2]['weight'] += 1
        else:
            # Thêm cạnh mới với trọng số 1
            G.add_edge(item1, item2, weight=1)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=20)

# Hiển thị trọng số trên các cạnh
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()