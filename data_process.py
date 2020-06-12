import datetime as dt
import os
from collections import OrderedDict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd
from lable import trace_id
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

imp_order_df = pd.read_csv("./data_processed/important_order.csv")
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

            if(cnt > 800000):
                cnt = 0
                order_id,rec = records.popitem(last=False)
                process_order_rec(order_event, order_id,rec)
                processed_order[order_id] = 1
        
    for order_id,rec in records.items():
        if order_id not in processed_order:
            process_order_rec(order_event, order_id,rec)

def process_order_rec(order_event, order_id, rec):
    trace = imp_order[order_id]

    trace = trace.split('-')
    for i, port1 in enumerate(trace):
        for port2 in trace[i+1:]:
            if '-'.join((port1,port2)) in trace_id:
                process_order_rec0(order_event, order_id, rec, port1, port2, i==0)
        

def get_departure_index(data, port):
    pos = port_pos[port]

    mae = np.sum(np.abs(data[:,(1,2)] - pos),axis=1)
    min_mae = np.min(mae)
    
    index = len(mae) - np.argmax(np.flip(mae < min_mae + 0.001))

    if(min_mae < 0.2):
        return index
    else:
        return None

def get_arrival_index(data, port):
    pos = port_pos[port]

    mae = np.sum(np.abs(data[:,(1,2)] - pos),axis=1)
    min_mae = np.min(mae)
    index = np.argmax(mae < min_mae + 0.1)

    if(min_mae < 0.2):
        return index + 1
    else:
        return None

def process_order_rec0(order_event, order_id, rec, departure, destination, is_departure_port_onboard_port):
    # rec shape:
    # time, longitude, latitude, speed, direction, nextport
    data = np.array(rec)

    departure_time = None
    arrival_time = None
    # try find departure_time and arrivel_time in order_event
    try:
        events = order_event[order_id]
        for (event_code,event_location,t) in events:
            if (event_code == shipment_onboard_date or event_code == transit_port_atd) and event_location == departure:
                departure_time = t
            elif (event_code == arrival_at_port or event_code == transit_port_ata) and event_location == destination:
                arrival_time = t
    except KeyError:
        None

    # trim data
    if arrival_time != None:
        arrival_index = np.argmax(data[:,0] < arrival_time)
    else:
        arrival_index = get_arrival_index(data, destination)
        if(arrival_index == None):
            print("[WRONG]Cannot find arrival index. Order id:", order_id,"Skip")
            return
        arrival_time = data[arrival_index - 1,0]
    
    if departure_time != None:
        departure_index = np.argmax(data[:,0] > departure_time)
    else :
        departure_index = get_departure_index(data, departure)
        if(departure_index == None):
            if(is_departure_port_onboard_port == True):
                departure_index = 0
                first_rec_time = data[0,0]
                first_rec_pos = data[0,(1,2)]
                dest_pos = port_pos[destination]
                depature_pos = port_pos[departure]

                dis_to_dest = geodistance(first_rec_pos,dest_pos)
                dis_to_depat = geodistance(first_rec_pos,depature_pos)

                departure_time = first_rec_time - (arrival_time - first_rec_time) * dis_to_depat / dis_to_dest
            else:
                print("[WRONG]Cannot find departure index. Order id:", order_id,"Skip")
                return
        else :
            departure_time = data[departure_index,0]

    data = data[departure_index:arrival_index]
    
    eta = (arrival_time - departure_time).total_seconds()
    #? is the first gps record time close to real onboard time ?
    travel_time = np.array([(t - departure_time).total_seconds() for t in data[:,0]]).reshape((-1,1))

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
