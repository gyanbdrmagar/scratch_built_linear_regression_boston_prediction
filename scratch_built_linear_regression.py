import pandas as pd
import matplotlib.pyplot as plt


file_path = 'Boston_House_Price_Data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

print(data.head())

df = data[['DIS','PRICE']]

# we try to plot the DIS: weighted distances to ﬁve Boston employment centers vs PRICE

#plt.scatter(df.DIS, df.PRICE)
#plt.show()

def loss_function(m,b,data_points):

    total_error = 0
    for i in range(len(data_points)):
        x = data_points.iloc[i].DIS
        y = data_points.iloc[i].PRICE

        total_error += (y - (m * x + b)) ** 2

    total_error / float(len(data_points))

    
def gradient_desent(m_now, b_now, data_points, L):

    m_gradient = 0
    b_gradient = 0

    n = len(data_points)

    for i in range(n):
        x = data_points.iloc[i].DIS
        y = data_points.iloc[i].PRICE

        m_gradient += -(2/n) * x *(y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L 
    b = b_now - b_gradient * L
    
    return m,  b


m = 0
b = 0
L = 0.0001
epoch = 400

for i in range(epoch):
    if i % 50 ==0:
        print(f'Epoch: {i}')
    m, b = gradient_desent(m,b,df, L)

print(m,b)

print('final model result plot')
plt.scatter(data.DIS, data.PRICE, color='blue')
plt.plot(list(range(0,30)), [m*x+b for x in range(0,30)], color = 'red')
plt.show()



