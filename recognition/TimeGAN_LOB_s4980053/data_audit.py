import pandas as pd

ob = pd.read_csv("data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv", header=None)
msg = pd.read_csv("data/AMZN_2012-06-21_34200000_57600000_message_10.csv", header=None)

print("rows:", len(ob), "| columns:", ob.shape[1])            # expect 40 columns
print("message rows:", len(msg))                               # should match orderbook rows

# Column 0 of the message file = time in seconds after midnight
start, end = msg[0].min() / 3600, msg[0].max() / 3600
print(f"time range: {start:.2f}h to {end:.2f}h")               # expect ~9.5 to 16.0

# Orderbook columns repeat per level: ask price, ask size, bid price, bid size
ask1, bid1 = ob[0], ob[2]
print("crossed books in real data:", (bid1 >= ask1).sum())     # should be 0
print("spread stats (price units):\n", (ask1 - bid1).describe())

print("\nfirst 3 rows:\n", ob.head(3))

# Empty levels are filled with dummy prices (readme: ±9999999999)
dummy = ((ob == 9999999999) | (ob == -9999999999)).sum().sum()
print("dummy price entries:", dummy)                           # hope for 0

# Trading halts are message type 7
print("trading halt messages:", (msg[1] == 7).sum())           # hope for 0

types = {1: "new order", 2: "partial cancel", 3: "delete",
         4: "visible trade", 5: "hidden trade", 7: "halt"}
side = {1: "BUY", -1: "SELL"}

for i in range(1, 11):
    t, typ, _, size, price, d = msg.iloc[i]
    changed = ob.columns[ob.iloc[i] != ob.iloc[i - 1]].tolist()
    print(f"{i}: {types[int(typ)]:14} {side[int(d)]:4} {int(size):5} @ ${price/10000:.2f}"
          f"  -> columns changed: {changed}")