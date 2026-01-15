import pandas as pd
from prophet import Prophet

#importando dados

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')


#treinando modelo

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast.to_excel('testando.xlsx')


#criando gráfico

fig1 = m.plot(forecast)
fig1.savefig('fig1.png')

fig2 = m.plot_components(forecast)
fig2.savefig('fig2.png')