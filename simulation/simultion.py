import os, sys
import random
from torch import tensor, device
from torch import nn

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from model.PredictionBasedGeneralModel import PredictionBasedGeneralModel
import traci
import matplotlib.pyplot as plt

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/sepehr/PycharmProjects/DAD/simulation/config/simulation.xml", "--seed", '1']
#
# model = PredictionBasedGeneralModel.load_from_checkpoint(
#     "/home/sepehr/PycharmProjects/DAD/lightning_logs/version_3/checkpoints/LSTMEncoderLSTM--v_num=00-epoch=03-validation_loss=0.00137-train_loss=0.00159.ckpt"
# ).cuda()
traci.start(sumoCmd)
step = 0
f = open('../dataset/v0.5/normal.output', 'w')
anm = ''
x = []
y = []
s = []
p = [[], [], [], [], []]
id = 0
max_speed = 0
flag = False
loss_function = nn.MSELoss()
ple = -1
while step < 100000:
    traci.simulationStep()
    if len(traci.vehicle.getIDList()) == 5:

        if traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger1':
            max_speed = 13.89 * 1
        elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger2':
            max_speed = 13.89 * 0.9
        elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger3':
            max_speed = 13.89 * 0.8
        elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger4':
            max_speed = 13.89 * 0.7
        elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger5':
            max_speed = 13.89 * 0.6

        l = ''
        for i in traci.vehicle.getIDList():
            l += str(traci.vehicle.getSpeed(i)) + ',' + str(traci.vehicle.getPosition(i)[0]) + ',' + str(
                traci.vehicle.getPosition(i)[1]) + ',' + str(traci.vehicle.getLaneIndex(i)) + ' '
        # l += str(id if flag else -1) + '\n'
        l += str(max_speed) + '\n'
        f.write(l)
        anm = ''
        s.append([traci.vehicle.getSpeed(traci.vehicle.getIDList()[i]) for i in range(0, 5)])
        x.append([(traci.vehicle.getPosition(traci.vehicle.getIDList()[i])[0] + 41.12) / (965.07 + 41.12) for i in
                  range(0, 5)])
        y.append([(traci.vehicle.getPosition(traci.vehicle.getIDList()[i])[1] - 44.20) / (50.60 - 44.20) for i in
                  range(0, 5)])

    # if len(s) >= 6 and len(traci.vehicle.getIDList()) == 5:
    #     i = len(s) - 6
    #     t = [[[s[j][k] / max(max(s)) for k in range(0, len(s[j]))], x[j], y[j]] for j in range(i, i + 5)]
    #     t = tensor(t, device=device('cuda'))
    #     t = model(t)
    #     for j in range(0, 5):
    #         p[j].append(t[0][0][0][j].item())
    # for i in range(0, 5):
    #     e = loss_function(t, tensor(s[len(s) - 1], device=device('cuda:0')))
    #     ids = traci.vehicle.getIDList()
    #     if e > 0.00169999:
    #         traci.vehicle.highlight(ids[id], color=(255, 0, 0))
    # else:
    #     traci.vehicle.highlight(ids[id], color=(0, 255, 0))
    if step % 200 == 0:
        ple = -1
        flag = False
        f.write('\n')
        # plt.plot(range(0, len(d[id])), d[id])
        # for i in range(0, len(s) - 5):
        #     t = [[s[j], x[j], y[j]] for j in range(i, i + 5)]
        #     t = tensor(t, device=device('cuda'))
        #     t = model([t])
        # p.append(t[0][0][id].item() * 20)
        # plt.plot(range(5, len(p) + 5), p)
        # if len(s) > 0:
        #     for j in range(0, 5):
        #         plt.plot(range(0, len(s)), [s[i][j] / max(max(s)) for i in range(0, len(s))], alpha=0.3)
        #         plt.plot(range(5, len(p[j]) + 5), p[j], alpha=1)
        # plt.plot(range(0, len(s[0])), s[1], alpha=0.3)
        # plt.plot(range(0, len(s[0])), s[2], alpha=0.3)
        # plt.plot(range(0, len(s[0])), s[3], alpha=0.3)
        # plt.plot(range(0, len(s[0])), s[4], alpha=0.3)
        # # # #
        id = random.randint(0, 4)
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('Speed')
        # plt.show()
        p = [[], [], [], [], []]
        x = []
        y = []
        s = []
    # if len(traci.vehicle.getIDList()) == 5 and ple != int(((step % 100) - (step % 10)) / 10 % 5):
    #     ple = int(((step % 100) - (step % 10)) / 10 % 5)
    #     ids = traci.vehicle.getIDList()
    #     traci.vehicle.changeLane(ids[ple], (traci.vehicle.getLaneIndex(ids[id]) + 1 + random.randint(0, 1)) % 3, 2)
    # if step % 3 == 0 and len(traci.vehicle.getIDList()) == 5:
    #     ids = traci.vehicle.getIDList()
    #     traci.vehicle.setColor(ids[id], (255, 0, 0))
    #     traci.vehicle.changeLane(ids[id], (traci.vehicle.getLaneIndex(ids[id]) + 1 + random.randint(0, 1)) % 3, 2)
    #     # traci.vehicle.setSpeedFactor(ids[id], traci.vehicle.getSpeedFactor(ids[id]) + 0.1)
    #     flag = True
    # if step % 100 == 20 and len(traci.vehicle.getIDList()) == 5:
    #     ids = traci.vehicle.getIDList()
    #     traci.vehicle.setColor(ids[id], (255, 0, 0))
    #     traci.vehicle.setSpeedMode(ids[id], 0)
    #     traci.vehicle.setSpeedFactor(ids[id], traci.vehicle.getSpeedFactor(ids[id]) + 0.1)
    # # traci.vehicle.setSpeed(ids[id], traci.vehicle.getSpeed(ids[id]) + 3)
    step += 1
