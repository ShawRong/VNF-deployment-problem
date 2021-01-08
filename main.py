import numpy as np
import random
import math
import time
import sys
import copy

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


#因为只是测试算法的可行性，这里随机生成5-10个服务链，用于表示某个时间片内收到的用户请求。
def servicechainsgenerate(vexnum, VNFnum):
    chains = list()
    picknum = random.randint(5, 10)
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
            selectedAP = servicechain['APs']
            if selectedAP[0]['vex'] == APtarget['vex']:
                flowratesum += servicechain['flowrate']

    processingcapacity = APtarget['processing capacity']
    if processingcapacity - flowratesum != 0:
        delay = 1 / (processingcapacity - flowratesum)
    else:
        delay = sys.maxsize
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

def isgood(input):
    if input >= 0:
        return True
    else:
        return False

def assignment(servernum, startpoint, consumption, reusenum, path, servicechian, VNFs, A, vexs):
    deploymentmin = [-1 for i in range(len(servicechian['VNFs']))]
    Csmin = sys.maxsize
    Prmin = sys.maxsize
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
                deployment1,Cs1,Pr1 = assignment(servernum + 1, pointer, consumption1, reusenum1, path, servicechian, VNFs, A, vexs)
                deployment1[servernum] = pointer
            else:
                deployment1 = [-1 for i in range(len(servicechian['VNFs']))]
                deployment1[servernum] = pointer
                Cs1 = consumption1
                Pr1 = reusenum1
            vexs[path[pointer]]['resource capacity'] += VNFs[servicechian['VNFs'][servernum]]['capacity need']
            vexs[path[pointer]]['VNF instance'].pop(-1)
            if Csmin > Cs1:
                Csmin = Cs1
                deploymentmin = deployment1
                Prmin = Pr1
        elif A[servernum][pointer] == 1:
            consumption2 = consumption
            reusenum2 = reusenum + 1
            for t in range(len(vexs[path[pointer]]['VNF instance'])):
                instance = vexs[path[pointer]]['VNF instance'][t]
                if instance['type'] == servicechian['VNFs'][servernum]:
                    vexs[path[pointer]]['VNF instance'][t]['processing capacity'] -= servicechian['flowrate']
                    break
            if servernum < len(servicechian['VNFs']) - 1:
                deployment2, Cs2, Pr2 = assignment(servernum + 1, pointer, consumption2, reusenum2, path, servicechian, VNFs, A, vexs)
                deployment2[servernum] = pointer
            else:
                deployment2 = [-1 for i in range(len(servicechian['VNFs']))]
                deployment2[servernum] = pointer
                Cs2 = consumption2
                Pr2 = reusenum2
            for t in range(len(vexs[path[pointer]]['VNF instance'])):
                instance = vexs[path[pointer]]['VNF instance'][t]
                if instance['type'] == servicechian['VNFs'][servernum]:
                    vexs[path[pointer]]['VNF instance'][t]['processing capacity'] += servicechian['flowrate']
                    break
            if Csmin > Cs2:
                Csmin = Cs2
                deploymentmin = deployment2
                Prmin = Pr2
            break

    return deploymentmin,Csmin,Prmin





def PGA(servicechain,path,vexs,VNFs):
    #compute A
    A = init_A(servicechain,path,vexs,VNFs)
    deployment, Cs, Pr = assignment(0, 0, 0, 0, path, servicechain, VNFs, A, vexs)
    return deployment, Cs, Pr




def istherereeuse(vex, VNFtype):
    for instance in vex['VNF instance']:
        if VNFtype == instance['type']:
            return True
    return False


def servicechainisdone(servicechain,path,deployment,vexs,edgelist,VNFs):
    servicechain['done'] = True
    selectedAP = list()
    for AP in servicechain['APs']:
        if AP['vex'] == path[0]:
            selectedAP.append(AP)
            break
    servicechain['APs'] = selectedAP
    #更新vexs
    for i in range(len(servicechain['VNFs'])):
        VNF = servicechain['VNFs'][i]
        server = path[deployment[i]]
        if istherereeuse(vexs[server], VNF) == False:
            vexs[server]['resource capacity'] -= VNFs[VNF]['capacity need']
            vexs[server]['VNF instance'].append(VNFs[VNF])
    #更新edgelist
    for i in range(len(path) - 1):
        for edge in edgelist:
            edgefinding1 = (path[i],path[i + 1])
            edgefinding2 = (path[i + 1],path[i])
            if edge['edge'] == edgefinding1 or edge['edge'] == edgefinding2:
                edge['bandwidth'] -= servicechain['flowrate']



def handleservicechains(servicechains, edgelist, matrix, vexs, VNFs):
    key = ['path','deployment','consumption','reuse time']
    solutions = list()
    for servicechain in servicechains:
        P = CDFSA(edgelist,matrix,servicechain,servicechains)
        deploymentmin = [-1 for i in range(len(servicechain['VNFs']))]
        Csmin = sys.maxsize
        Prmin = 0
        pathmin = list()
        for path in P:
            deployment, Cs, Pr = PGA(servicechain, path, vexs, VNFs)

            if Csmin > Cs:
                deploymentmin = deployment
                Csmin = Cs
                Prmin = Pr
                pathmin = path

        solution = dict([(k, []) for k in key])
        solution['path'] = pathmin
        solution['deployment'] = deploymentmin
        solution['consumption'] = Csmin
        solution['reuse time'] = Prmin
        if Csmin != sys.maxsize :
            servicechainisdone(servicechain, pathmin, deploymentmin, vexs, edgelist, VNFs)
        solutions.append(solution)

    return solutions


def LCS(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return s

def advanced_Pathpriority(path, vexs, VNFs, flowrate):
    listofset = list()
    for server in path:
        tempset = set()
        for i in range(len(vexs[server]['VNF instance'])):
            if vexs[server]['VNF instance'][i]['processing capacity'] >= flowrate:
                tempset.add(vexs[server]['VNF instance'][i]['type'])
        listofset.append(tempset)
    listfromset = list()

    #to do LCS to find the longest length

    for i in range(len(path)):
        for VNF in VNFs:
            if VNF in listofset[i]:
               listfromset.append(VNF)
    #use LCS
    A = copy.deepcopy(listfromset)
    B = copy.deepcopy(VNFs)

    listofLCS = LCS(A, B)
    length = len(listofLCS)
    retlist = list()
    index = 0
    for i in range(len(path)):
        templist = list()
        while (index < len(listofLCS)) and (listofLCS[index] in listofset[i]):
            templist.append(listofLCS[index])
            index = index + 1
        retlist.append(templist)
    return length, retlist, listofLCS



def releaseInstance(vex, VNFtype):
    for i in range(len(vex['VNF instance'])):
        if vex['VNF instance'][i]['type'] == VNFtype:
            vex['resource capacity'] += vex['VNF instance'][i]['capacity need']
            vex['VNF instance'].pop(i)


def dealtypeN(VNFnum,startpoint,path, VNFchian, vexs, VNFs):
    deployment_final = [-1 for i in range(len(VNFchian))]
    for pointer in range(startpoint, len(path)):
        flag = False
        for instance in vexs[path[pointer]]['VNF instance']:
            if instance['type'] == VNFchian[VNFnum]:
                if instance['type'] == VNFchian[VNFnum]:
                    flag = True
        if vexs[path[pointer]]['resource capacity'] >= VNFs[VNFchian[VNFnum]]['capacity need'] and flag == False:
            vexs[path[pointer]]['resource capacity'] -= VNFs[VNFchian[VNFnum]]['capacity need']
            vexs[path[pointer]]['VNF instance'].append(VNFs[VNFchian[VNFnum]])
            if VNFnum < len(VNFchian) - 1:
                deployment = dealtypeN(VNFnum + 1, pointer, path, VNFchian, vexs, VNFs)
                deployment[VNFnum] = pointer
            else:
                deployment = [-1 for i in range(len(VNFchian))]
                deployment[VNFnum] = pointer
            vexs[path[pointer]]['resource capacity'] += VNFs[VNFchian[VNFnum]]['capacity need']
            vexs[path[pointer]]['VNF instance'].pop(-1)
            Status = True
            for i in range(VNFnum,len(deployment)):
                if deployment[i] == -1:
                    Status = False
                    break
            if Status == True:
                deployment_final = deployment
                break
    return deployment_final



def advanced_GA(servicechain,path,vexs,VNFs):
    deployment = [-1 for i in range(len(servicechain['VNFs']))]
    consumption = 0
    reusenum = 0
    length, lists, LCS = advanced_Pathpriority(path, vexs, servicechain['VNFs'], servicechain['flowrate'])
    typeR = set(LCS)
    print(servicechain['VNFs'],lists,path)
    #deal the reusable
    for VNFnum in range(len(servicechain['VNFs'])):
        VNFtype = servicechain['VNFs'][VNFnum]
        if VNFtype in typeR: #reusable
            for server in range(len(path)):
                if VNFtype in lists[server]:
                    deployment[VNFnum] = server
                    reusenum = reusenum + 1
                    for instance in vexs[path[server]]['VNF instance']:
                        if instance['type'] == VNFtype:
                            instance['processing capacity'] -= servicechain['flowrate']
                            break
                    break
    #print("deployment",deployment)
    #deal the not reusable
    #get VNF chain slice
    newVNFchain = list(servicechain['VNFs'])
    for i in range(len(newVNFchain)):
        for singlelist in lists:
            if newVNFchain[i] in singlelist and newVNFchain[i] == singlelist[-1]:
                newVNFchain[i] = -1
    for singlelist in lists:
        i = 0
        while i < len(newVNFchain):
            if newVNFchain[i] in singlelist:
                newVNFchain.pop(i)
                i = i - 1
            i = i + 1
    newVNFchain.append(-1)
    VNFchainslice = list()
    start = 0
    for i in range(len(newVNFchain)):
        if newVNFchain[i] == -1:
            if i == 0 and newVNFchain[start:i] == []:
                VNFchainslice.append(newVNFchain[start:i])
            if newVNFchain[start:i] != []:
                VNFchainslice.append(newVNFchain[start:i])
            start = i + 1

    #get path slice
    start = 0
    pathslice = list()
    pathstart = list()
    for i in range(len(path)):
        if lists[i] != []:
            pathslice.append(path[start:(i + 1)])
            pathstart.append(start)
            start = i
    pathslice.append(path[start:])
    pathstart.append(start)

    print(VNFchainslice, pathslice, pathstart)
    for i in range(len(VNFchainslice)):
        if VNFchainslice[i] != []:
            deploymentslice = dealtypeN(0,0,pathslice[i],VNFchainslice[i],vexs,VNFs)
            print(deploymentslice)
            index = 0
            while deployment[index] != -1:
                index = index + 1
            for item in deploymentslice:
                deployment[index] = item + pathstart[i]
                index = index + 1
            for VNFtype in VNFchainslice[i]:
                consumption += VNFs[VNFtype]['capacity need']
    if -1 in deployment:
        consumption = sys.maxsize
    print("deployment",deployment, consumption)
    return deployment, consumption, reusenum






def advanced_handleservicechains(servicechains, edgelist, matrix, vexs, VNFs):
    key = ['path','deployment','consumption','reuse time']
    solutions = list()
    for servicechain in servicechains:
        P = CDFSA(edgelist,matrix,servicechain,servicechains)
        deploymentmin = [-1 for i in range(len(servicechain['VNFs']))]
        Csmin = sys.maxsize
        Prmin = 0
        pathmin = list()
        priormax = 0
        for path in P:
            prior, _, _ = advanced_Pathpriority(path, vexs, servicechain['VNFs'], servicechain['flowrate'])
            if priormax > prior and Csmin != sys.maxsize:
                continue
            priormax = prior
            deployment, Cs, Pr = advanced_GA(servicechain, path, vexs, VNFs)
            if Csmin > Cs:
                deploymentmin = deployment
                Csmin = Cs
                Prmin = Pr
                pathmin = path


        solution = dict([(k, []) for k in key])
        solution['path'] = pathmin
        solution['deployment'] = deploymentmin
        solution['consumption'] = Csmin
        solution['reuse time'] = Prmin
        if -1 not in deploymentmin :
            servicechainisdone(servicechain, pathmin, deploymentmin, vexs, edgelist, VNFs)
        solutions.append(solution)

    return solutions

def advanced_handleservicechains_sub(servicechains, edgelist, matrix, vexs, VNFs):
    key = ['path','deployment','consumption','reuse time']
    solutions = list()
    for servicechain in servicechains:
        P = CDFSA(edgelist,matrix,servicechain,servicechains)
        deploymentmin = [-1 for i in range(len(servicechain['VNFs']))]
        Csmin = sys.maxsize
        Prmin = 0
        pathmin = list()
        priormax = 0
        for path in P:
            prior, _, _ = advanced_Pathpriority(path, vexs, servicechain['VNFs'], servicechain['flowrate'])
            if priormax > prior and Csmin != sys.maxsize:
                continue
            priormax = prior
            deployment, Cs, Pr = PGA(servicechain, path, vexs, VNFs)
            if Csmin > Cs:
                deploymentmin = deployment
                Csmin = Cs
                Prmin = Pr
                pathmin = path


        solution = dict([(k, []) for k in key])
        solution['path'] = pathmin
        solution['deployment'] = deploymentmin
        solution['consumption'] = Csmin
        solution['reuse time'] = Prmin
        if Csmin != sys.maxsize :
            servicechainisdone(servicechain, pathmin, deploymentmin, vexs, edgelist, VNFs)
        solutions.append(solution)

    return solutions

if __name__ == '__main__':
    vexs, edgelist, servicechains, matrix, VNFs = init("layer42.txt")
    vexs_sub = copy.deepcopy(vexs)
    edgelist_sub = copy.deepcopy(edgelist)
    servicechains_sub = copy.deepcopy(servicechains)
    matrix_sub = copy.deepcopy(matrix)
    VNFs_sub = copy.deepcopy(VNFs)

    print(vexs)
    print(edgelist)
    print(servicechains)
    print(matrix)
    print(VNFs)

    testchains = list()
    testchains.append(servicechains[0])
    time_start = time.time()
    solutions = advanced_handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    consumptionsum1 = sys.maxsize
    for i in range(len(solutions)):
        if solutions[i]['consumption'] != sys.maxsize:
            consumptionsum1 += solutions[i]['consumption']
    print(testchains)
    print(solutions)
    print(consumptionsum1)
    print(time_end - time_start)
    for vex in vexs:
        if vex['resource capacity'] < 0:
            print("error")
    for edge in edgelist :
        if edge['bandwidth'] < 0 :
            print("error")

    testchains_sub = list()
    testchains_sub.append(servicechains[0])
    time_start = time.time()
    solutions_sub = advanced_handleservicechains_sub(testchains_sub, edgelist_sub, matrix_sub, vexs_sub, VNFs_sub)
    time_end = time.time()
    consumptionsum1_sub = sys.maxsize
    for i in range(len(solutions_sub)):
        if solutions_sub[i]['consumption'] != sys.maxsize:
            consumptionsum1 += solutions_sub[i]['consumption']
    print(testchains_sub)
    print(solutions_sub)
    print(consumptionsum1_sub)
    print(time_end - time_start)
    for vex in vexs_sub:
        if vex['resource capacity'] < 0:
            print("error")
    for edge in edgelist_sub:
        if edge['bandwidth'] < 0:
            print("error")

'''
    time_start = time.time()
    solutions_sub = advanced_handleservicechains_sub(servicechains_sub, edgelist_sub, matrix_sub, vexs_sub, VNFs_sub)
    time_end = time.time()
    consumptionsum2 = 0
    for i in range(len(solutions_sub)):
        if solutions_sub[i]['consumption'] != sys.maxsize:
            consumptionsum2 += solutions_sub[i]['consumption']
    print(consumptionsum2)
    print(time_end - time_start)
    for vex in vexs_sub:
        if vex['resource capacity'] < 0:
            print("error")
    for edge in edgelist_sub :
        if edge['bandwidth'] < 0 :
            print("error")
'''

'''
    testchains = list()
    testchains.append(servicechains[0])
    time_start = time.time()
    solutions = advanced_handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    print(solutions[0])
    print(time_end - time_start)

    testchains_sub = list()
    testchains_sub.append(servicechains_sub[0])
    time_start = time.time()
    solutions_sub = handleservicechains(testchains_sub, edgelist_sub, matrix_sub, vexs_sub, VNFs_sub)
    time_end = time.time()
    print(solutions_sub[0])
    print(time_end - time_start)
'''
'''
    time_start = time.time()
    solutions = advanced_handleservicechains(servicechains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    for i in range(len(servicechains)):
        print(servicechains[i])
        print(solutions[i])
    print(time_end - time_start)


    print(vexs_sub)
    print(edgelist_sub)
    print(servicechains_sub)
    print(matrix_sub)
    print(VNFs_sub)
    time_start = time.time()
    solutions_sub = handleservicechains(servicechains_sub, edgelist_sub, matrix_sub, vexs_sub, VNFs_sub)
    time_end = time.time()
    for i in range(len(servicechains_sub)):
        print(servicechains_sub[i])
        print(solutions_sub[i])
    print(time_end - time_start)
'''
'''
    testchains = list()
    testchains.append(servicechains[0])
    print("begin")
    print(servicechains[0])
    print(vexs)
    for edge in edgelist:
        print(edge)
    time_start = time.time()
    solutions = advanced_handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    print("end")
    print(solutions[0])
    print(servicechains[0])
    path = solutions[0]['path']
    deployment = solutions[0]['deployment']
    for i in range(len(deployment)):
        index = path[deployment[i]]
        print(vexs[index]['resource capacity'])
        print(vexs[index]['VNF instance'])
    for edge in edgelist :
        print(edge)
    print(time_end - time_start)
'''

'''
    time_start = time.time()
    solutions = handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    print(solutions[0])
    print(time_end - time_start)
'''
'''
    P = CDFSA(edgelist,matrix,servicechains[0],servicechains)
    deployment,_,_ =  PGA(servicechains[0],P[0],vexs,VNFs)
    print(P[0])
    print(servicechains[0]['VNFs'])
    print(deployment)
'''
'''
    testchains = list()
    testchains.append(servicechains[0])

    solutions = advanced_handleservicechains(servicechains, edgelist, matrix, vexs, VNFs)
    for i in range(len(servicechains)):
        print(servicechains[i])
        print(solutions[i])
'''

'''
    time_start = time.time()
    solutions = advanced_handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    print(solutions[0])
    print(time_end - time_start)

    time_start = time.time()
    solutions = handleservicechains(testchains, edgelist, matrix, vexs, VNFs)
    time_end = time.time()
    print(solutions[0])
    print(time_end - time_start)
'''

'''
        P = CDFSA(edgelist,matrix,servicechains[0],servicechains)
        cnt = advanced_PathCheck(P[0], vexs, servicechains[0]['VNFs'])
        for server in P[0] :
            print("server ",server)
            for instance in vexs[server]['VNF instance']:
                print(instance['type'])
        print(servicechains[0]['VNFs'])
        print(cnt)
'''


