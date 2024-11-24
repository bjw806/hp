import csv
from binance.client import Client

client = Client()


csvfile = open('BTCUSDT-2024-2024-15m.csv', 'w', newline='')
candlestick_writer = csv.writer(csvfile, delimiter=',')


candlesticks = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "1 Jan, 2024", "23 Nov, 2024")

for candlestick in candlesticks:
    candlestick[0] = candlestick[0] / 1000 # divide timestamp to ignore miliseconds
    print(candlestick[0])
    candlestick_writer.writerow(candlestick)


csvfile.close()