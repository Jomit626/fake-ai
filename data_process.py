import datetime as dt
import os
from collections import OrderedDict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

csv_file_port = r"./dataset/port.csv"
csv_file_train = r"./dataset/train0523.csv"
csv_file_order_event = r"./dataset/loadingOrderEvent.csv"

result_csv_folder = r"./data_processed/orders"
if not os.path.exists(result_csv_folder):
    os.mkdir(result_csv_folder)

shipment_onboard_date = 0   # "SHIPMENT ONBOARD DATE"
transit_port_ata = 1        # "TRANSIT PORT ATA"
transit_port_atd = 2        # "TRANSIT PORT ATD"
arrival_at_port = 3         # "ARRIVAL AT PORT"

imp_order_df = pd.read_csv("./data_processed/imp_order.csv")
imp_order = dict(zip(imp_order_df["OrderId"],imp_order_df["Trace"]))

port_df = pd.read_csv(csv_file_port)
port_pos = dict(zip(port_df["TRANS_NODE_NAME"], np.array(port_df.loc[:,("LONGITUDE","LATITUDE")])))

def geodistance(pos1,pos2):
    lng1,lat1 = pos1
    lng2,lat2 = pos2
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def load_order_event(filename):
    order_event = {}
    with open(filename, "r") as f:
        f.readline()    #ignore first line
        while True:
            line = f.readline().strip()
            if(line[0] == ','):
                break

            (order_id, event_code, event_location, date_time) = line.split(',')

            if order_id not in imp_order:
                continue

            if event_code == "SHIPMENT ONBOARD DATE":
                event_code = shipment_onboard_date
            elif event_code == "TRANSIT PORT ATA":
                event_code = transit_port_ata
            elif event_code == "TRANSIT PORT ATD":
                event_code = transit_port_atd
            elif event_code == "ARRIVAL AT PORT":
                event_code = arrival_at_port
            else:
                continue
            events = order_event.get(order_id)
            if(events == None):
                events = []
                order_event[order_id] = events
            try:
                t = datetime.strptime(date_time, r"%Y/%m/%d %H:%M")
            except ValueError as identifier:
                t = datetime.strptime(date_time, r"%Y/%m/%d")
            events.append((event_code,event_location,t))

    for k,events in order_event.items():
        events.sort(key= lambda x : x[2])
    
    return order_event
    

def load_gps_rec(filename, order_event):
    records = OrderedDict()
    processed_order = {}
    with open(filename, "r") as f:
        cnt = 0
        while True:
            line = f.readline()[:-1]
            if not line:
                break
            data = [text.strip('"') for text in line.split(",")]
            order_id = data[0]

            if order_id not in imp_order:
                continue

            cnt += 1
            rec = records.get(order_id)
            if(rec == None):
                if order_id in processed_order:
                    print("Warning!",order_id)
                    exit()
                rec = []
                records[order_id] = rec
            
            rec.append([
                datetime.strptime(data[2], r"%Y-%m-%dT%H:%M:%S.%fZ"),   # time
                float(data[3]), # longitude
                float(data[4]), # latitude
                float(data[6]), # speed
                float(data[7]), # direction
                data[8]         # nextport
            ])

            if(cnt > 1000000):
                cnt = 0
                order_id,rec = records.popitem(last=False)
                process_order_rec(order_event, order_id,rec)
                processed_order[order_id] = 1
        
    for order_id,rec in records.items():
        process_order_rec(order_event, order_id,rec)

def process_order_rec(order_event, order_id,rec):
    # rec shape:
    # time, longitude, latitude, speed, direction, nextport
    data = np.array(rec)
    trace = imp_order[order_id]

    # assume first two
    departure ,destination = trace.split('-')[:2]

    onboard_time = None
    arrival_time = None
    # trim data
    dest_pos = port_pos[destination]
    mse = np.sum(np.square(data[:,(1,2)] - dest_pos),axis=1)

    min_mse = np.min(mse)
    index = np.argmax(mse < min_mse*1.1)
    if(index == 0):
        index = -1
    # try find arrivel_time in order_event
    try:
        events = order_event[order_id]
        for (event_code,event_location,t) in events:
            if event_code == shipment_onboard_date and event_location == departure:
                onboard_time = t
            elif ((event_code == arrival_at_port or event_code == transit_port_ata) and event_location == destination):
                arrival_time = t
    except KeyError:
        onboard_time = None
        arrival_time = None
    # if not found
    if(arrival_time == None):
        #arrival_time = data[-1,0]   # use last gps record as arrival_time
        arrival_time = data[index,0]
    if(onboard_time == None):
        first_rec_time = data[0,0]
        first_rec_pos = data[0,(1,2)]
        depature_pos = port_pos[departure]

        dis_to_dest = geodistance(first_rec_pos,dest_pos)
        dis_to_depat = geodistance(first_rec_pos,depature_pos)

        onboard_time = first_rec_time - (arrival_time - first_rec_time) * dis_to_depat / (dis_to_dest + dis_to_depat)

        
    data = data[:index + 1]

    eta = (arrival_time - onboard_time).total_seconds()
    #? is the first gps record time close to real onboard time ?
    travel_time = np.array([(t - onboard_time).total_seconds() for t in data[:,0]]).reshape((-1,1))

    trace = '-'.join((departure ,destination))

    data = np.concatenate((
        np.array([order_id for _ in range(len(data))]).reshape((-1,1)),
        travel_time,
        data[:,1:5],
        np.array([trace for _ in range(len(data))]).reshape((-1,1)),
        np.array([eta for _ in range(len(data))]).reshape((-1,1)),
    ),axis=1)

    pd.DataFrame(data).to_csv(os.path.join(result_csv_folder, trace + '_' + order_id + ".csv"),header=False,index=False)

print("preprocess done.")
order_event = load_order_event(csv_file_order_event)
print("load order_event done.")
load_gps_rec(csv_file_train,order_event)
print("done.")
