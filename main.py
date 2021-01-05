import numpy as np
import random
import math

# 读取txt文件，得到邻接矩阵
def readtxt(filename):
    f = open(filename, "r")
    line = f.readline()
    line = line[:-1]
    cnt = 0
    vertexnum = 0
    edges = list()
    while line:
        if cnt == 0:
            vertexnum = int(line)
            cnt = 1
        else:
            edge = line.split(' ', 1)
            edge[0] = int(edge[0]) - 1
            edge[1] = int(edge[1]) - 1
            edge = tuple(edge)
            edges.append(edge)
        line = f.readline()
        line = line[:-1]
    f.close()
    relation_matrix = [[0 for i in range(vertexnum)] for i in range(vertexnum)]
    for (x, y) in edges:
        relation_matrix[x][y] = 1
        relation_matrix[y][x] = 1
    return relation_matrix,edges

#随机化生成节点之间的延迟
def addlatency(matrix,min = 0,max = 30):
    for i in range(len(matrix)):
        for j in range(i,len(matrix[i])):
            if i == j :
                continue
            if matrix[i][j] == 1:
                matrix[i][j] = random.randint(min,max)
                matrix[j][i] = matrix[i][j]
    return matrix

#生成节点list
def vexgenerate(vexnum):
    vexs = list()
    key = ["id","resource capacity","VNF instance"]
    for i in range(vexnum):
        vex = dict([(k, []) for k in key])
        vexs.append(vex)
    return vexs

#随机化生成节点上的resource capacity。
def vexcapacity(vexs):
    cnt = 0
    for vex in vexs:
        vex['id'] = cnt
        vex['resource capacity'] = random.randint(0,200)
        cnt = cnt + 1
    return vexs

#随机生成其上的VNF实例
def vexVNF(vexs,VNFs):


    for vex in vexs:
        picknum = random.randint(0, 8)
        pickVNFs = random.sample(range(len(VNFs)), picknum)
        VNFchain = []
        for pickVNF in pickVNFs:
            new = VNFs[pickVNF]
            VNFchain.append(new)
        vex['VNF instance'] = VNFchain

    return vexs

#随机生成边及其带宽参数
def edgegenerate(edges):
    edgelist = list()
    key = ["edge","bandwidth"]
    for t in edges:
        edge = dict([(k, []) for k in key])
        edge['edge'] = t
        edge['bandwidth'] = random.randint(0,1000)
        edgelist.append(edge)
    return edgelist


#随机生成服务链相关
def servicechiangenerate(vexnum, VNFnum):

    #generate APs
    APs = list()
    key = ["vex", "processing capacity"]

    times = random.randint(1, 5)
    randlist = random.sample(range(vexnum), times)
    for i in range(times):
        AP = dict([(k, []) for k in key])
        AP['vex'] = randlist[i]
        AP['processing capacity'] = random.randint(100, 200)
        APs.append(AP)

    #generate destination
    destination = random.randint(0, vexnum - 1)

    #generate flow rate
    flowrate = random.randint(30, 60)

    #generate latency deadline
    latencydeadline = random.randint(30, 80)

    #generate VNF chain
    picknum = random.randint(1, 6)
    VNFchain = random.sample(range(VNFnum), picknum)

    chain = {'done': False,'APs': APs, 'destination': destination, 'flowrate': flowrate, 'latency deadline': latencydeadline, 'VNFs': VNFchain}
    return chain


#因为只是测试算法的可行性，这里随机生成10-100个服务链，用于表示某个时间片内收到的用户请求。
def servicechainsgenerate(vexnum, VNFnum):
    chains = list()
    picknum = random.randint(10, 100)
    for i in range(picknum):
        chain = servicechiangenerate(vexnum,VNFnum)
        chains.append(chain)

    return chains


def VNFsgenerate():
    VNFs = list()
    key = ["type","capacity need", "processing capacity"]
    cnt = 0
    for i in range(20):
        VNF = dict([(k, []) for k in key])
        VNF['type'] = cnt
        VNF['capacity need'] = random.randint(20,50)
        VNF['processing capacity'] = random.randint(100,200)
        VNFs.append(VNF)
        cnt = cnt + 1
    return VNFs


def init(filename):
    VNFs = VNFsgenerate()
    matrix, edges = readtxt(filename)
    matrix = addlatency(matrix)

    vexnum = len(matrix)
    VNFnum = len(VNFs)
    vexs = vexgenerate(vexnum)
    vexs = vexcapacity(vexs)
    vexs = vexVNF(vexs, VNFs)
    edgelist = edgegenerate(edges)
    servicechains = servicechainsgenerate(vexnum, VNFnum)

    return vexs,edgelist,servicechains,matrix,VNFs


def init_M(edgelist,flowrate,vexnum):
    matrix = [[0 for i in range(vexnum)] for i in range(vexnum)]
    for edge in edgelist:
        bandwidth = edge['bandwidth']
        (x, y) = edge['edge']
        if int(bandwidth / flowrate) > 1 :
            matrix[x][y] = 2
            matrix[y][x] = 2
        elif int(bandwidth / flowrate) == 1:
            matrix[x][y] = 1
            matrix[y][x] = 1

    return matrix




def init_E(matrix):
    return matrix



def queuingdelaygenerate(servicechains, APtarget):
    flowratesum = 0
    for servicechain in servicechains:
        if servicechain['done'] == True:
            AP = servicechain['APs']
            if AP['vex'] == APtarget['vex']:
                flowratesum += servicechain['flowrate']

    processingcapacity = APtarget['processing capacity']
    delay = 1 / (processingcapacity - flowratesum)
    return delay

def selection_path(startpoint,destination,latency,latencylimitaion,stack,P,vexnum,M,E):
    stack.append(startpoint)
    if startpoint == destination :
        new = stack[:]
        P.append(new)
        stack.pop(-1)
        return
    for v in range(vexnum):
        if M[startpoint][v] > 0 and (latency + E[startpoint][v]) < latencylimitaion :
            if M[startpoint][v] == 2 :
                M[startpoint][v] = 0
                selection_path(v, destination, latency + E[startpoint][v], latencylimitaion, stack, P, vexnum, M, E)
                M[startpoint][v] = 2
            elif M[startpoint][v] == 1:
                M[startpoint][v] = 0
                M[v][startpoint] = 0
                selection_path(v, destination, latency + E[startpoint][v], latencylimitaion, stack, P, vexnum, M, E)
                M[startpoint][v] = 1
                M[v][startpoint] = 1
    stack.pop(-1)




def CDFSA(edgelist,matrix,servicechain,servicechains):
    M = init_M(edgelist,servicechain['flowrate'],len(matrix))
    E = init_E(matrix)
    P = list()
    stack = []
    for AP in servicechain['APs']:
        u = AP['vex']
        queuingdelay = queuingdelaygenerate(servicechains,AP)
        if queuingdelay < 0:
            continue
        selection_path(u,servicechain['destination'],queuingdelay,servicechain['latency deadline'],stack,P,len(matrix),M,E)

    return P


def init_A(servicechain,path,vexs,VNFs):
    matrix = [[0 for i in range(len(path))] for i in range(len(servicechain['VNFs']))]
    count = 0
    for VNF in servicechain['VNFs']:
        cnt = 0
        for server in path:
            for instance in vexs[server]['VNF instance']:
                if VNF == instance['type'] and instance['processing capacity'] - servicechain['flowrate'] >= 0:
                    matrix[count][cnt] = 1
            cnt = cnt + 1
        count = count + 1
    return matrix


def assignment(servernum, startpoint, consumption, reusenum, path, servicechian, VNFs, A, vexs):
    deploymentmin = [-1 for i in range(len(servicechian['VNFs']))]
    Csmin = -1
    Prmin = -1
    for pointer in range(startpoint, len(path)):
        flag = False
        for instance in vexs[path[pointer]]['VNF instance']:
            if instance['type'] == VNFs[servicechian['VNFs'][servernum]]['type']:
                flag = True
        if A[servernum][pointer] == 0 and vexs[path[pointer]]['resource capacity'] >= VNFs[servicechian['VNFs'][servernum]]['capacity need'] and flag == False:
            consumption1 = consumption + VNFs[servicechian['VNFs'][servernum]]['capacity need']
            reusenum1 = reusenum
            vexs[path[pointer]]['resource capacity'] -= VNFs[servicechian['VNFs'][servernum]]['capacity need']
            vexs[path[pointer]]['VNF instance'].append(VNFs[servicechian['VNFs'][servernum]])
            if servernum < len(servicechian['VNFs']) - 1:
                deployment,Cs,Pr = assignment(servernum + 1, pointer, consumption1, reusenum1, path, servicechian, VNFs, A, vexs)
                deployment[servernum] = pointer
            else:
                deployment = [-1 for i in range(len(servicechian['VNFs']))]
                deployment[servernum] = pointer
                Cs = consumption1
                Pr = reusenum1
            vexs[path[pointer]]['resource capacity'] += VNFs[servicechian['VNFs'][servernum]]['capacity need']
            vexs[path[pointer]]['VNF instance'].pop(-1)
            if Csmin != -1 and Csmin > Cs:
                Csmin = Cs
                deploymentmin = deployment
                Prmin = Pr
            elif Csmin == -1 :
                Csmin = Cs
                deploymentmin = deployment
                Prmin = Pr
        elif A[servernum][pointer] == 1:
            consumption2 = consumption
            reusenum2 = reusenum + 1
            if servernum < len(servicechian['VNFs']) - 1:
                deployment, Cs, Pr = assignment(servernum + 1, startpoint, consumption2, reusenum2, path, servicechian, VNFs, A, vexs)
                deployment[servernum] = pointer
            else:
                deployment = [-1 for i in range(len(servicechian['VNFs']))]
                deployment[servernum] = pointer
                Cs = consumption2
                Pr = reusenum2
            if Csmin != -1 and Csmin > Cs:
                Csmin = Cs
                deploymentmin = deployment
                Prmin = Pr
            elif Csmin == -1:
                Csmin = Cs
                deploymentmin = deployment
                Prmin = Pr
            break

    return deploymentmin,Csmin,Prmin





def PGA(servicechain,path,vexs,VNFs):
    #compute A
    A = init_A(servicechain,path,vexs,VNFs)
    deployment, Cs, Pr = assignment(0, 0, 0, 0, path, servicechain, VNFs, A, vexs)
    return deployment, Cs, Pr


def handleservicechains(servicechains, edgelist, matrix, vexs, VNFs):
    key = ['path','deployment','consumption','reuse time']
    solutions = list()
    for servicechain in servicechains:
        P = CDFSA(edgelist,matrix,servicechain,servicechains)
        deploymentmin = deploymentmin = [-1 for i in range(len(servicechain['VNFs']))]
        Csmin = -1
        Prmin = -1
        pathmin = list()
        for path in P:
            deployment, Cs, Pr = PGA(servicechain, path, vexs, VNFs)
            if Csmin != -1 and Csmin > Cs:
                deploymentmin = deployment
                Csmin = Cs
                Prmin = Pr
                pathmin = path
            elif Csmin == -1:
                deploymentmin = deployment
                Csmin = Cs
                Prmin = Pr
                pathmin = path

        solution = dict([(k, []) for k in key])
        solution['path'] = pathmin
        solution['deployment'] = deploymentmin
        solution['consumption'] = Csmin
        solution['reuse time'] = Prmin
        solutions.append(solution)

    return solutions

if __name__ == '__main__':
    vexs, edgelist, servicechains, matrix, VNFs = init("bellsouth.txt")
    print(vexs)
    print(edgelist)
    print(servicechains)
    print(matrix)
    print(VNFs)

    solutions = handleservicechains(servicechains,edgelist, matrix, vexs, VNFs)
    for i in range(len(servicechains)) :
        print(servicechains[i])
        print(solutions[i])