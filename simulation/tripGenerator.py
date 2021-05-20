import random

# streets = ['gneE6', 'gneE16', 'gneE15']
g = [5, 10, 15]
f = open("./calgary/osm.passenger.trips.xml", "w")
# f = open("./training/trips1.xml", "w")
k = 0
streets = [1, 3, 5, 7]
f.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
f.write("xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n")
# normal classes
f.write(
    "<vType id=\"veh_passenger1\" vClass=\"passenger\" speedFactor=\"normc(1,0.1,0.9,1.1)\" speedDev=\"0.1\""
    " lcCooperative=\"1\" lcKeepRight=\"0\"/>\n")
f.write(
    "<vType id=\"veh_passenger2\" vClass=\"passenger\" speedFactor=\"normc(0.8,0.1,0.7,0.9)\" speedDev=\"0.1\"/>\n")
f.write(
    "<vType id=\"veh_passenger3\" vClass=\"passenger\" speedFactor=\"normc(0.7,0.1,0.6,0.8)\" speedDev=\"0.1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger2\" vClass=\"passenger\" speedFactor=\"1.0\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger3\" vClass=\"passenger\" speedFactor=\"1.1\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger4\" vClass=\"passenger\" speedFactor=\"1.06\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger5\" vClass=\"passenger\" speedFactor=\"1.08\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")

f.write(
    "<vType id=\"abnormal1\" vClass=\"passenger\"  speedFactor=\"normc(1.25,0.1,1.2,1.3)\" speedDev=\"0.1\" accel=\"7\" decel=\"8\" minGap=\"0.5\" color=\"1,0,0\" sigma=\"0.05\" maxSpeed=\"140\" tau=\"1\" "
    "lcStrategic=\"1\" lcSpeedGain=\"1\" lcKeepRight=\"0\" lcOpposite=\"1\"/>\n")

f.write(
    "<vType id=\"abnormal2\" vClass=\"passenger\"  speedFactor=\"normc(0.75,0.1,0.7,0.8)\" speedDev=\"0.1\" accel=\"7\" decel=\"8\" minGap=\"0.5\" color=\"1,0,0\" sigma=\"0.05\" maxSpeed=\"140\" tau=\"1\"/>\n")

for i in range(0, 360000, 300):
    a = random.randint(0, 12)
    c = random.randint(1, 3)

    for j in range(0, 5):
        cl = f'veh_passenger{1}'
        _from = 'calgary1'
        _to = 'calgary2'
        if j == a:
            cl = 'abnormal1'
        f.write(
            f'<trip id=\"veh{k * 10 + j}\"  type=\"{cl}\" depart=\"{i + j * 3}.00\"  departLane=\"random\" departSpeed=\"max\" from=\"{_from}\" to=\"{_to}\" />\n')
    k += 1

f.write('</routes>')
