import pandas as pd
import numpy as np
csv_file_port = r"./dataset/port.csv"
csv_file_train = r"./dataset/train0523.csv"
csv_file_order_event = r"./dataset/loadingOrderEvent.csv"
csv_file_test_data = r"./dataset/A_testData0531.csv"

imp_order_file = r"./data_processed/important_order.csv"


train_trace = {}
# exract trace in train data
with open(csv_file_train, "r") as f:
    while True:
        line = f.readline()[:-1]
        if not line:
            break

        data = [text.strip('"') for text in line.split(",")]
        order_id = data[0]
        trace = data[-1]

        if trace:
            train_trace[order_id] = trace

test_data_df = pd.read_csv(csv_file_test_data)
test_trace = test_data_df["TRANSPORT_TRACE"].unique()
test_trace = dict(zip(test_trace,range(len(test_trace))))

imp_order = []
for order_id,trace in train_trace.items():
    traces = trace.split('-')
    for i, port1 in enumerate(traces):
        for port2 in traces[i+1:]:
            if '-'.join((port1,port2)) in test_trace:
                imp_order.append((order_id,trace))
                break

pd.DataFrame(imp_order,columns=["OrderId","Trace"]).to_csv(imp_order_file,index=False)