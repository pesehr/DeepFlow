import os, sys
import random
import tripGenerator
from torch import nn
import math
from model.Siamese import Siamese
from torch import tensor, device

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import matplotlib.pyplot as plt


class Simulator:

    def __init__(self, address='../dataset/test/memorial-dec', w=True):
        self.step = 0
        self.end = 100000
        self.vehicles = 5
        sumoBinary = "sumo-gui"
        sumoCmd = [sumoBinary, "-c", "/home/sepehr/PycharmProjects/DAD/simulation/training/simulation2.xml", "--seed",
                   '1']
        traci.start(sumoCmd)

        self.id = 0
        self.flag = False
        self.w = w
        if self.w:
            self.file = open(address, 'w')
            self.s = [[], [], [], [], [], []]
            self.x = [[], [], [], [], [], []]
            self.y = [[], [], [], [], [], []]
            self.t = ['', '', '', '', '']
        # self.model = Siamese.load_from_checkpoint(
        #     "../checkpoint/LSTMEncoderLSTM--v_num=00-epoch=00-validation_loss=0.53419-train_loss=0.00000.ckpt"
        # ).cuda()
        self.round = -1
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.max_speed = 0

        self.min_x = 100000
        self.max_x = -10000
        self.min_y = 100000
        self.max_y = -10000

        self.run()

    def init_variables(self):
        if self.step % 300 == 0:
            if self.w:
                self.write()
            self.round += 1
            r = random.randint(0, 10)
            # if r == 1 or r == 3 or r == 5:
            self.id = 0
            # else:
            # self.id = -1
            self.s = [[], [], [], [], [], []]
            self.x = [[], [], [], [], [], []]
            self.y = [[], [], [], [], [], []]
            self.t = ['', '', '', '', '']
            self.max_speed = 0
            self.flag = False

    def write(self):
        for i in range(0, self.vehicles):
            l = ''
            for j in range(2, len(self.x[i])):
                # dx = self.x[i][j] - self.x[i][j - 1]
                # dx2 = self.x[i][j - 1] - self.x[i][j - 2]
                # l += f'{round(self.x[i][j], 3)},{round(dx, 4)},{round(dx2 - dx, 4)},{round(self.y[i][j], 3)},{round(self.s[i][j], 2)} '
                l += f'{round(self.x[i][j], 5)},{round(self.y[i][j], 5)} '
            plt.plot(range(0, len(self.s[i])), self.s[i], label=f'Vehicle #{i + 1}')
            self.file.write(l)
            self.file.write(';')
            if 'abnormal' in self.t[i]:
                self.file.write('1')
            else:
                self.file.write('-1')
            self.file.write('\n')
        plt.legend()
        # plt.ylim([-1, 10])
        plt.xlabel("time (s)")
        plt.ylabel("Speed (m/s)")
        # plt.show()

    def log(self):
        # if self.step == self.end - 1:
        #     print(f'{self.min_x} {self.max_x} {self.min_y} {self.max_y}')
        for i in range(0, self.vehicles):
            id = f'veh{self.round * 10 + i}'
            if id in traci.vehicle.getIDList():
                x = traci.vehicle.getPosition(id)[0]
                y = traci.vehicle.getPosition(id)[1]
                s = traci.vehicle.getSpeed(id)

                self.t[i] = traci.vehicle.getTypeID(id)
                self.x[i].append(self.normalize(x, 2331.069758335272, 331.6160350313303)) #memorial
                self.y[i].append(self.normalize(y, 1557.7461908018008, 736.1227529386199))

                # self.x[i].append(self.normalize(x, 3065.339689905422, 2772.2621071169474)) #edmonton
                # self.y[i].append(self.normalize(y, 2131.654558026798, 1363.8048687718222))

                # self.x[i].append(self.normalize(x, 599.9004098356795, -195.0734))  # train
                # self.y[i].append(self.normalize(y, 292.0, 298.4))

                # if x < self.min_x:
                #     self.min_x = x
                # if x > self.max_x:
                #     self.max_x = x
                # if y < self.min_y:
                #     self.min_y = y
                # if y > self.max_y:
                #     self.max_y = y

                # self.x[i].append(-(2 * ((x - 545.945) / (2152.780 - 545.945)) - 1))  # 9ave
                # self.y[i].append(2 * ((y - 449.112) / (464.02 - 449.112)) - 1)
                if s > 0.001:
                    self.s[i].append(s)
                else:
                    self.s[i].append(0)

    def over_speed(self):
        if self.step % 300 == 50 and len(traci.vehicle.getIDList()) == self.vehicles:
            ids = traci.vehicle.getIDList()
            traci.vehicle.setColor(ids[self.id], (255, 0, 0))
            # traci.vehicle.setSpeedMode(ids[0], 0)
            traci.vehicle.setSpeedFactor(ids[0], traci.vehicle.getSpeedFactor(ids[0]) + 0.2)
            traci.vehicle.setSpeedFactor(ids[1], traci.vehicle.getSpeedFactor(ids[1]) + 0.2)
            traci.vehicle.setSpeedFactor(ids[2], traci.vehicle.getSpeedFactor(ids[2]) + 0.2)
            traci.vehicle.setSpeedFactor(ids[3], traci.vehicle.getSpeedFactor(ids[3]) + 0.2)
            traci.vehicle.setSpeedFactor(ids[4], traci.vehicle.getSpeedFactor(ids[4]) + 0.2)
            self.flag = True

    def speed_class(self):
        if self.step % 300 == 20 and len(traci.vehicle.getIDList()) == self.vehicles:
            ids = traci.vehicle.getIDList()
            traci.vehicle.setSpeedMode(ids[0], 0)
            traci.vehicle.setColor(ids[self.id], (255, 0, 0))
            self.flag = True

    def under_speed(self):
        if self.step % 300 == 20 and len(traci.vehicle.getIDList()) == self.vehicles:
            ids = traci.vehicle.getIDList()
            traci.vehicle.setColor(ids[self.id], (255, 0, 0))
            # traci.vehicle.setSpeedMode(ids[0], 0)
            traci.vehicle.setSpeedFactor(ids[0], traci.vehicle.getSpeedFactor(ids[self.id]) - 0.3)
            self.flag = True

    def lane_anomaly(self):
        if len(traci.vehicle.getIDList()) == self.vehicles:
            self.flag = True
            ids = traci.vehicle.getIDList()
            traci.vehicle.changeLane(ids[self.id],
                                     (traci.vehicle.getLaneIndex(ids[self.id]) + 1 + random.randint(0, 1)) % 3,
                                     2)

    def lane_change(self):
        if len(traci.vehicle.getIDList()) == self.vehicles:
            ids = traci.vehicle.getIDList()
            traci.vehicle.changeLane(ids[self.id],
                                     (traci.vehicle.getLaneIndex(ids[self.id]) + 1 + random.randint(0, 1)) % 3, 2)
        # ple = int(((self.step % 100) - (self.step % 10)) / 10)
        # if len(traci.vehicle.getIDList()) == self.vehicles and ple < self.vehicles and self.step % 10 == 0:
        #     ids = traci.vehicle.getIDList()
        #     traci.vehicle.changeLane(ids[ple], (traci.vehicle.getLaneIndex(ids[ple]) + 1 + random.randint(0, 1)) % 3, 2)

    def normalize(self, d, max, min):
        return 2 * ((d - min) / (max - min)) - 1

    def run(self):
        while self.step < self.end:
            self.init_variables()
            traci.simulationStep()
            # self.lane_change()
            # self.no_stop()
            # self.over_speed()
            # self.speed_class()
            # r = random.randint(0, 10)
            # if r == 2 or r == 4 or r == 6:
            # self.over_speed()
            # for id in traci.vehicle.getIDList():
            #     if 'abnormal' in traci.vehicle.getTypeID(id):
            #         traci.vehicle.setSpeedMode(id, 0)
            # if r == 1:
            # elif r == 3:
            #     self.speed_class()
            # if r == 2:
            #     self.speed_class()
            # else:
            # self.speed_class()
            # self.detect()
            self.log()
            self.step += 1


Simulator()
