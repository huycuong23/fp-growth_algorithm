# importing modules
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# load dataset
import csv

# Open the CSV file
file = str(input("nhap ten file: "))
dataset = pd.read_csv(file)

# print dataset shape
print("Hình dạng của dữ liệu:\n")
print(dataset.shape)
print("DONE!\n")
# display first few rows of the dataset


print("in các cột và một vài hàng bằng cách sử dụng head: \n")
print(dataset.head())
print("DONE!\n")
# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])
transaction = np.array(transaction)
print("in các mảng giao dịch: \n")
print(transaction)
print("DONE!\n")

# Transform them into a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"])
df["incident_count"] = 1

# Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan"].index
df.drop(indexNames, inplace=True)

# Making a New Appropriate Pandas DataFrame for Visualizations
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

# Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Blues')
print("top 5 mặt hàng được khách hàng mua nhiều nhất.: \n")
print(df_table)
print("DONE! \n")


# Add a column for treemap
df_table["all"] = "Top 50 items"

# Create treemap using plotly
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(50), hover_data=['items'],
                  color_continuous_scale='Blues')

# Plot the treemap
fig.show()

# Transform every transaction to a separate list & gather them into Numpy Array


import pandas as pd
import csv

# Open the CSV file
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

# In kết quả
print("Chúng ta cần chuyển đổi tập dữ liệu của mình thành giá trị đúng và sai. Ví dụ: nếu giao dịch chứa một mục, chúng tôi sẽ điền đúng và nếu không có giao dịch, chúng tôi sẽ điền sai. \n")
print(encoded_df)
print("DONE! \n")


min_supportInput = float(input("nhap 0<= min support <= 1: "))
res = fpgrowth(encoded_df, min_support=min_supportInput, use_colnames=True)
print("Tập dữ liệu của chúng tôi hiện đã sẵn sàng và chúng tôi có thể triển khai thuật toán tăng trưởng FP để tìm các mục xuất hiện thường xuyên bằng cách đặt mức hỗ trợ tối thiểu thành " + str(min_supportInput) + "\n")
print(res.head(200))
print(" DONE! \n")

if len(res.head(200)) > 0:
    res = association_rules(res, metric="lift", min_threshold=1)
    print("Bảng dữ liệu!\n")
    print(res)
    print("DONE!\n")
else:
    print("Không có mặt hàng nào thoả mãn độ hỗ trợ " + str(min_supportInput))


if len(res.head(200)) > 0:
    print(" Bảng dữ liệu sắp xếp theo confidence! \n")
    print(res.sort_values("confidence",ascending=False))
    print(" DONE! \n")