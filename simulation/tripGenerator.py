import random

# streets = ['gneE6', 'gneE16', 'gneE15']
g = [5, 10, 15]
# f = open("./calgary/osm.passenger.trips.xml", "w")
f = open("./training/trips1.xml", "w")
k = 0
streets = [1, 3, 5, 7]
f.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
f.write("xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n")
# normal classes
f.write(
    "<vType id=\"veh_passenger1\" vClass=\"passenger\" speedFactor=\"normc(1,0.5,0.9,1.1)\" speedDev=\"0.1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger2\" vClass=\"passenger\" speedFactor=\"1.0\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger3\" vClass=\"passenger\" speedFactor=\"1.1\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger4\" vClass=\"passenger\" speedFactor=\"1.06\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")
# f.write(
#     "<vType id=\"veh_passenger5\" vClass=\"passenger\" speedFactor=\"1.08\" speedDev=\"0.0\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")

f.write(
    "<vType id=\"abnormal1\" vClass=\"passenger\"  speedFactor=\"1.2\" lcKeepRight=\"0\" lcCooperative=\"1\" lcSpeedGain=\"1\" lcStrategic=\"1\"/>\n")

for i in range(0, 360000, 300):
    a = random.randint(1, 15)
    for j in range(0, 5):
        # c = random.randint(1, 3)
        cl = f'veh_passenger{1}'
        if j == a:
            cl = 'abnormal1'
        f.write(
            f'<trip id=\"veh{k * 10 + j}\"  type=\"{cl}\" depart=\"{i + j * 3}.00\"  departLane=\"random\" departSpeed=\"max\" from=\"gneE2\" to=\"gneE3\"/>\n')
    k += 1

f.write('</routes>')
