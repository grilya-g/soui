import json
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data = pd.read_excel('Данные v2.xlsx', sheet_name='Бр_дневка - 3 (основной)')
    data = data.rename(
        columns={'дата': 'date', 'направление': 'direction', 'выход': 'close'}
    )
    data['date'] = pd.to_datetime(data['date'])
    data['direction'] = data['direction'].str.replace('ш', '0').str.replace('л', '1').astype(int)

    pred = pd.read_excel('Данные v2.xlsx', sheet_name='Прогноз')
    pred['дата'].describe()


    # Загрузка данных
    data = data.rename(columns = {'date': 'ds', 'close': 'y'})

    # Инициализация и обучение модели Prophet
    model = Prophet()
    model.fit(data)

    # Создание DataFrame для будущих дат
    future = model.make_future_dataframe(periods=60)  # Прогноз на 60 дней вперед

    # Прогнозирование
    forecast = model.predict(future)
    final_pred = pd.merge(pred.rename(columns={'дата': 'ds'}), forecast[['ds', 'yhat']], on='ds', how='left')

    final_pred.loc[:, 'направление'] = (final_pred['yhat'].diff() > 0).astype(int)
    final_pred['направление'] = final_pred['направление'].astype(int)
    final_pred.head()

    final_pred.loc[final_pred['yhat'].isna()]
    final_pred['направление'].to_json('forecast_class.json', orient='records')

