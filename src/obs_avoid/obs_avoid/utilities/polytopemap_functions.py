# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:00:02 2020

@author: Dr. Ngoc Thinh Nguyen

Library of functions for polytope map

"""


import numpy as np
from scipy.optimize import linprog
import pypoman
from rdp import rdp as rdp
import gdspy
from poly_decomp import poly_decomp as pd
import cv2 as cv
from shapely.geometry import LineString

def distance(point1, point2):
    return np.sqrt((point1[0]- point2[0])**2 + (point1[1]- point2[1])**2)

def check_point_in_between(point, pointA, pointB):
    epsilon = 1e-5
    return abs(distance(point,pointA) + distance(point,pointB) - distance(pointA,pointB)) < epsilon

def line(p,q):
    # return ax + by = c passing 2 points p, q
    [xp, yp] = p
    [xq, yq] = q
    a = yp - yq
    b = xq - xp
    c = xp * (yp - yq) - yp * (xp - xq)
    return a, b, c

def distance_point2line(r,p,q):
    # return distance from point r to line passing through p,q
    (a,b,c) = line(p,q)
    return abs(a*r[0]+b*r[1]-c)/np.sqrt(a**2+b**2)

def vertices2halfspace(vertices):
    # return halfspace representation of the (convex) polytope, defined by its vertices
    A = []
    B = []
    epsilon = 1e-5
    for i in range(len(vertices) - 1):
        (a, b, c) = line(vertices[i], vertices[i+1])
        j = i-1
        while abs(a * vertices[j][0] + b * vertices[j][1] - c)<epsilon:
            j = j - 1
        if ((a * vertices[j][0] + b * vertices[j][1] - c) < 0):
            sign = 1
        else:
            sign = -1
        A.append([sign * a, sign * b])
        B.append(sign * c)
    (a, b, c) = line(vertices[0], vertices[-1])
    j = -2
    while abs(a * vertices[j][0] + b * vertices[j][1] - c)<epsilon:
        j = j - 1
    if ((a * vertices[j][0] + b * vertices[j][1] - c) < 0):
        sign = 1
    else:
        sign = -1
    A.append([sign * a, sign * b])
    B.append(sign * c)
    return A, B

def del_halfspace(A,B,p1,p2):
    A_copy = A[:]
    B_copy = B[:]
    ## checks if two points are on one of the hyperplanes and deletes the corresponding halfspace from (A,B)
    for i in range(len(A)):
        # print(i,len(A))
        a=A[i]
        b=B[i]
        # check if both points on halfspace
        if abs(a[0]*p1[0]+a[1]*p1[1]-b)<1e-07 and abs(a[0]*p2[0]+a[1]*p2[1]-b)<1e-07:
            A_copy.remove(a)
            B_copy.remove(b)
            break
    return A_copy, B_copy

def vertices2halfspace_remove_edge(vertices, m, n):
    # return halfspace representation of the (convex) polytope, defined by its vertices
    # not including the edge of vertices[m] and vertices[n]
    A = []
    B = []
    epsilon = 1e-5
    for i in range(len(vertices) - 1):
        if (i!=min(m,n)) or ((i+1)!=max(m,n)):
            (a, b, c) = line(vertices[i], vertices[i+1])
            j = i-1
            while abs(float(a * vertices[j][0] + b * vertices[j][1] - c))<epsilon:
                j = j - 1
            if ((a * vertices[j][0] + b * vertices[j][1] - c) < 0):
                sign = 1
            else:
                sign = -1
            A.append([sign * a, sign * b])
            B.append(sign * c)
            #print('Step ', i, ': ', A, B)
    if (min(m,n)!=0) or (max(m,n)!=len(vertices) - 1):
        (a, b, c) = line(vertices[0], vertices[-1])
        j = -2
        while abs(a * vertices[j][0] + b * vertices[j][1] - c)<epsilon:
            j = j - 1
        if ((a * vertices[j][0] + b * vertices[j][1] - c) < 0):
            sign = 1
        else:
            sign = -1
        A.append([sign * a, sign * b])
        B.append(sign * c)
    return A, B

def transition_zone(poly1, poly2):
    #calculates safe transition zone between two polytopes as another polytope and returns the half space
    #assumes only two common vertices
    share_ver=shared_vertices(poly1, poly2)
    (A1_mod, B1_mod)=vertices2halfspace_remove_edge(poly1,share_ver[0][0],share_ver[1][0])
    (A2_mod, B2_mod)=vertices2halfspace_remove_edge(poly2,share_ver[0][1],share_ver[1][1])
    A= A1_mod + A2_mod
    B= B1_mod + B2_mod

    # A,B here is ready to use but we can simplify them further. Not very useful.
    # vertices1=[poly1[share_ver[0][0]], poly1[share_ver[1][0]]]
    # B = np.array([B])
    # trans_zone = np.append(A,-B.T, axis=1)
    # medium_point = np.array([(vertices1[0][0]+ vertices1[1][0])/2, (vertices1[0][1]+ vertices1[1][1])/2 ])
    # #medium_point = np.array([0.5, 0.5])
    # simplified_trans_zone = HalfspaceIntersection(trans_zone, medium_point)
    # halfspace_trans_zone = ConvexHull(simplified_trans_zone.intersections)
    # halfspace_trans_zone = halfspace_trans_zone.equations
    # A_matrix = halfspace_trans_zone[:, 0:2]
    # B_matrix = - halfspace_trans_zone[:,-1]
    # return A_matrix, B_matrix
    return A, B

def transition_zone_splitting(poly1, poly2):
    #calculates safe transition zone between two polytopes as another polytope and returns the half space
    #assumes only two common vertices
    share_ver=shared_vertices(poly1, poly2)
    #vertices1=[poly1[share_ver[0][0]], poly1[share_ver[1][0]]]
    #vertices2=[poly2[share_ver[0][1]], poly2[share_ver[1][1]]]
    #print('V1: ', vertices1)
    #print('V2: ', vertices2)
    #print(share_ver)
    (A1, B1)=vertices2halfspace(poly1)
    (A2, B2)=vertices2halfspace(poly2)
    (A1_mod, B1_mod)=vertices2halfspace_remove_edge(poly1,share_ver[0][0],share_ver[1][0])
    (A2_mod, B2_mod)=vertices2halfspace_remove_edge(poly2,share_ver[0][1],share_ver[1][1])
    #(A1_mod, B1_mod)=del_halfspace(A1,B1,vertices1[0],vertices1[1])
    #(A2_mod, B2_mod)=del_halfspace(A2,B2,vertices1[0],vertices1[1])
    #print(A1,B1,A1_mod, B1_mod)
    A_first=np.concatenate((A1, A2_mod))
    B_first=np.concatenate((B1, B2_mod))
    A_second=np.concatenate((A1_mod, A2))
    B_second=np.concatenate((B1_mod, B2))
    return [[A_first, B_first], [A_second, B_second]]


def internal_point_optimize(A,B):
    # find 1 point in the intermidate of polytope Ax<=B (strictly inside)
    # required by scipy
    # efficient
    B = np.array([B])
    halfspaces = np.append(A,-B.T, axis=1)
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),(halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    point = res.x[0:2]
    return point

def internal_point(A,B, epsilon=0.5):
    # find 1 point in the intermidate of polytope Ax<=B
    # required by scipy
    c = [1, 0]
    B_new = [(b-epsilon) for b in B]
    res = linprog(c, A_ub=A, b_ub=B_new, bounds=(None, None))
    while not res.success:
        epsilon = epsilon/10
        B_new = [(b-epsilon) for b in B]
        res = linprog(c, A_ub=A, b_ub=B_new, bounds=(None, None))
    point = res.x[0:2]
    return point

def vertices_from_halfspace(A,B):
    #medium_point = internal_point(A,B)
    #B = np.array([B])
    #halfspace = np.append(A,-B.T, axis=1)
    #polygon = HalfspaceIntersection(halfspace, medium_point)
    #vertices = polygon.intersections
    #AB = polygon.equations
    #A_matrix = AB[:, 0:2]
    #B_matrix = - AB[:,-1]
    vertices = pypoman.compute_polytope_vertices(np.array(A), np.array(B))
    return [i.tolist() for i in vertices] #, A_matrix, B_matrix

def enlarge_zone(poly1, poly2, give_vertices=False):
    # calculates the enlarged zone from poly1 to poly2 (poly1 + transition zone from poly1 to poly2)
    share_ver=shared_vertices(poly1, poly2)
    vertices1=[poly1[share_ver[0][0]], poly1[share_ver[1][0]]]
    (A, B) = transition_zone(poly1, poly2)
    # medium_point = internal_point(A,B)
    # B = np.array([B])
    # trans_zone = np.append(A,-B.T, axis=1)
    # simplified_trans_zone = HalfspaceIntersection(trans_zone, medium_point)
    # vertices_trans_zone = simplified_trans_zone.intersections
    vertices_trans_zone = pypoman.compute_polytope_vertices(np.array(A), np.array(B))

    #print(vertices_trans_zone)
    enlarge_zoned = poly1[:]
    for vertices in vertices_trans_zone:
        #if check_point_zone(poly2, vertices):
        if (vertices[0]-vertices1[0][0])**2+ (vertices[1]-vertices1[0][1])**2>0.001:
            if (vertices[0]-vertices1[1][0])**2 + (vertices[1]-vertices1[1][1])**2>0.001:
                enlarge_zoned.append([vertices[0], vertices[1]])
    #convexhull_enlarge_zone = ConvexHull(enlarge_zoned)
    #halfspace_enlarge_zone = convexhull_enlarge_zone.equations
    #A_matrix = halfspace_enlarge_zone[:, 0:2]
    #B_matrix = - halfspace_enlarge_zone[:,-1]
    (A_matrix, B_matrix) = pypoman.duality.compute_polytope_halfspaces(enlarge_zoned)
    if give_vertices:
        #vertices_index = convexhull_enlarge_zone.vertices
        #vertices = [enlarge_zoned[i] for i in vertices_index]
        vertices = pypoman.compute_polytope_vertices(A_matrix, B_matrix)
        return A_matrix, B_matrix, vertices
    else:
        return A_matrix, B_matrix



def findzone(ws, pos):
    # find if any zone in ws contains pos
    nzone = len(ws)
    n = 0
    find = False
    nfind = -1
    while n < nzone and not find:
        x_de = [ws[n][k][0] for k in range(len(ws[n]))]
        y_de = [ws[n][k][1] for k in range(len(ws[n]))]
        #vertices = np.array(ws)
        #print(vertices)
        #x_de = vertices[:,0]
        #y_de = vertices[:,1]
        if min(x_de)<= pos[0] and pos[0] <= max(x_de):
            if  min(y_de)<= pos[1] and pos[1] <= max(y_de):
                # (A_ws,B_ws) = vertices2halfspace(ws[n])
                # find = True
                # for i in range(len(B_ws)):
                #     lhs = A_ws[i][0] * pos[0] + A_ws[i][1] * pos[1]
                #     if lhs > B_ws[i]:
                #         find = False
                find = check_point_zone(ws[n], pos)
                if find:
                    nfind = n
        n = n + 1
    return nfind

def check_point_zone(zone, pos):
    # (A_ws,B_ws) = vertices2halfspace(zone)
    # in_zone = True
    # for i in range(len(B_ws)):
    #     lhs = A_ws[i][0] * pos[0] + A_ws[i][1] * pos[1]
    #     if lhs > B_ws[i]:
    #         in_zone = False
    (A,B) = pypoman.duality.compute_polytope_halfspaces(zone)

    return ((np.dot(A, pos) - np.array(B))<=0).all()


def distance_point2zone(r, zone):
    # provide a relative distance from a point to a zone
    inZone = check_point_zone(zone, r) #findzone([zone], r)
    if not inZone:
        distance = []
        for i in range(len(zone)):
            distance.append(np.sqrt((r[0]-zone[i][0])**2+ (r[1]-zone[i][1])**2))
        distance2vertices = min(distance)
        midpoint = mid_point(zone)
        distance2midpoint = np.sqrt((r[0]-midpoint[0])**2+ (r[1]-midpoint[1])**2)
        res = (0.8 * distance2vertices + 0.2 * distance2midpoint)/2
    else:
        res = 0.0
    return res

def shared_vertices(zone0, zone1):
    # provide the common vertices between two polytopes
    # result = [[number of point 1 w.r.t. zone 1, number of point 1 w.r.t. zone 2]...]
    epsilon = 1e-5
    nvertices = []
    for n in range(len(zone0)):
        for i in range(len(zone1)):
            if abs(zone0[n][0]-zone1[i][0])<epsilon and abs(zone0[n][1]-zone1[i][1])<epsilon:
                nvertices.append([n,i])
            if len(nvertices) == 2:
                break
        if len(nvertices) == 2:
                break
    return nvertices

def connect_pairs(ws):
    # find all the pairs of two connected polytopes in workspace ws
    res = []
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i<j:
                shareedge = shared_vertices(ws[i], ws[j])
                if len(shareedge)==2:
                    res.append([[i,j], shareedge[0], shareedge[1]])
    return res

def gen_map(ws, nzone0):
    # give the whole connected map of the workspace to zone 0
    pairs = connect_pairs(ws)
    npair = len(pairs)
    res = []
    pair_remove = []
    for i in range(npair):
        if pairs[i][0][0]==nzone0:
            res.append([nzone0, pairs[i][0][1]])
            pair_remove.append(i)
        elif pairs[i][0][1]==nzone0:
            res.append([nzone0, pairs[i][0][0]])
            pair_remove.append(i)
    listpair = np.delete(np.arange(npair), pair_remove)
    ncount = 0
    while len(listpair)> 0 and ncount<npair:
        newway = []
        for j in range(len(res)):
            nzone_now = res[j][-1]
            for i in listpair:
                if pairs[i][0][0]==nzone_now:
                    if nzone_now==res[j][-1]:

                        res[j].append(pairs[i][0][1])
                    else:
                        newway_tmp = res[j][:]
                        newway_tmp.pop(len(newway_tmp)-1)
                        newway_tmp.append(pairs[i][0][1])
                        newway.append(newway_tmp)
                    pair_remove.append(i)
                elif pairs[i][0][1]==nzone_now:
                    if nzone_now==res[j][-1]:
                        res[j].append(pairs[i][0][0])
                    else:
                        newway_tmp = res[j][:]
                        newway_tmp.pop(len(newway_tmp)-1)
                        newway_tmp.append(pairs[i][0][0])
                        newway.append(newway_tmp)
                    pair_remove.append(i)
        for k in range(len(newway)):
            res.append(newway[k])
        listpair = np.delete(np.arange(npair), pair_remove)
        ncount = ncount + 1
    return res

def disconnectedzone(ws, nzone0=-1):
    if nzone0 == -1:
        roadmap = gen_map(ws, 0)
    else:
        roadmap = gen_map(ws, nzone0)
    # roadmap = gen_map(ws, 46) # 46 is the starting region of lab 2 map
    res = []
    for i in range(len(ws)):
        found = False
        j = 0
        while j<len(roadmap) and not found:
            if np.any(np.array(roadmap[j])==i):
                found = True
            j = j + 1
        if not found:
            res.append(i)
    return res

def gen_way(ws, nzone0, nzone1):
    # generate a list of consecutive zones connecting nzone0 and nzone1
    roadmap = gen_map(ws, nzone0)
    way = []
    found = False
    n = 0
    while not found and n < len(roadmap):
        road = roadmap[n]
        for i in range(len(road)):
            if road[i]==nzone1:
                way = road[:i+1]
                found = True
        n = n +1
    return way



def gate_of_way(way, ws):
    # give the vertices connecting two consecutive polytopes in a way
    res = []
    for i in range(len(way)-1):
        shareedge = shared_vertices(ws[way[i]], ws[way[i+1]])
        if len(shareedge)==2:
            res.append([shareedge[0][0], shareedge[1][0]])
        else:
            res.append([])
    return res

def find_suitable_zone(ws, ref, give_in_zone=False, corrected=False, nzone0 = -1):
    # to find the closest zone to the reference point
    # corrected = True means you are sure that all workspaces are connected
    # give_in_zone = True to have (new_zone, original_zone)
    # 1) find if any zone in ws contains pos
    nzone_origin = findzone(ws, ref)
    nzone = nzone_origin
    if not corrected:
        restricted_zone = disconnectedzone(ws, nzone0 = nzone0)
        if nzone==-1 or np.any(np.array(restricted_zone)==nzone):
            distance_zone_ref = []
            index_zone = []
            for i in range(len(ws)):
                if not np.any(np.array(restricted_zone)==i):
                    distance_zone_ref.append(distance_point2zone(ref, ws[i]))
                    index_zone.append(i)
            nzone = index_zone[distance_zone_ref.index(min(distance_zone_ref))]
    else:
        if nzone==-1:
            distance_zone_ref = []
            for i in range(len(ws)):
                #if not np.any(np.array(restricted_zone)==nzone):
                distance_zone_ref.append(distance_point2zone(ref, ws[i]))
            nzone = distance_zone_ref.index(min(distance_zone_ref))
    if give_in_zone:
        return nzone, nzone_origin
    else:
        return nzone

def find_reference(ws, pos0, ref):
    # find the list of middle points of the edge connecting two zones
    # error found: disconnected zone -> gen_way gives [] -> error -- solved
    nzone = find_suitable_zone(ws, ref)
    nzone0 = findzone(ws, pos0)
    way2zone = gen_way(ws, nzone0, nzone)
    middle_point = []
    if len(way2zone)>1:
        gate = gate_of_way(way2zone, ws)
        for i in range(len(gate)):
            point_1 = ws[way2zone[i]][gate[i][0]]
            point_2 = ws[way2zone[i]][gate[i][1]]
            middle_point.append([(point_1[0]+point_2[0])/2, (point_1[1]+point_2[1])/2])
    else:
        middle_point.append(ref)
    return middle_point

def plot_ws(ws, ax, col):
    # plot the whole workspace in ax with color col
    for i in range(len(ws)):
        x_de = [ws[i][k][0] for k in range(len(ws[i]))]
        x_de.append(ws[i][0][0])
        y_de = [ws[i][k][1] for k in range(len(ws[i]))]
        y_de.append(ws[i][0][1])
        ax.plot(x_de, y_de, color = col)
    return 0

def plot_road(ws, road, ax, col):
    # plot the list of zones, defining by road.
    for i in road:
        x_de = [ws[i][k][0] for k in range(len(ws[i]))]
        x_de.append(ws[i][0][0])
        y_de = [ws[i][k][1] for k in range(len(ws[i]))]
        y_de.append(ws[i][0][1])
        ax.plot(x_de, y_de, color = col)
    return 0

def mid_point(zone):
    # find the middle point of a zone
    for i in range(len(zone)):
        x_de = [zone[k][0] for k in range(len(zone))]
        y_de = [zone[k][1] for k in range(len(zone))]
    x = sum(x_de)/ len(zone)
    y = sum(y_de)/ len(zone)
    return [x, y]

def plot_line_road(ws, road, gate_way, ax, colway):
    # plot the line connecting the middle points of zone and edges
    (x0, y0) = mid_point(ws[road[0]])
    x_line = [x0]
    y_line =[y0]
    for i in range(len(road)-1):
        nzone = road[i]
        x_midpoint = (ws[nzone][gate_way[i][0]][0] + ws[nzone][gate_way[i][1]][0])/2
        y_midpoint = (ws[nzone][gate_way[i][0]][1] + ws[nzone][gate_way[i][1]][1])/2
        x_line.append(x_midpoint)
        y_line.append(y_midpoint)
    (x_end, y_end) = mid_point(ws[road[-1]])
    x_line.append(x_end)
    y_line.append(y_end)
    ax.plot(x_line, y_line, color = colway)
    ax.scatter(x_line, y_line, color = colway)
    return 0

def distance_line_road(ws, road, pos0, ref):
    gate_way = gate_of_way(road, ws)
    # give the line connecting the middle points of zone and edges
    #(x0, y0) = mid_point(ws[road[0]])
    x_line = [pos0[0]]
    y_line =[pos0[1]]
    distance = 0
    for i in range(len(road)-1):
        nzone = road[i]
        x_midpoint = (ws[nzone][gate_way[i][0]][0] + ws[nzone][gate_way[i][1]][0])/2
        y_midpoint = (ws[nzone][gate_way[i][0]][1] + ws[nzone][gate_way[i][1]][1])/2
        distance += np.sqrt((x_line[-1]-x_midpoint)**2+ (y_line[-1]-y_midpoint)**2)
        x_line.append(x_midpoint)
        y_line.append(y_midpoint)
    if check_point_zone(ws[road[-1]], ref):
        distance += np.sqrt((x_line[-1]-ref[0])**2+ (y_line[-1]-ref[1])**2)
    else:
        (x_end, y_end) = mid_point(ws[road[-1]])
        distance += np.sqrt((x_line[-1]-x_end)**2+ (y_line[-1]-y_end)**2)
    #x_line.append(x_end)
    #y_line.append(y_end)
    return distance




def gen_way_with_minimal_distance(ws, pos0, ref, give_zone=False, nzone_start=0, nzone_end=0):
    if not give_zone:
        nzone0 = findzone(ws, pos0)
        nzone1 = find_suitable_zone(ws, ref)
    else:
        nzone0 = nzone_start
        nzone1 = nzone_end
    # generate a list of consecutive zones connecting nzone0 and nzone1
    roadmap = gen_map(ws, nzone0)

    way_list = []
    for n in range(len(roadmap)):
        road = roadmap[n]
        for i in range(len(road)):
            if road[i]==nzone1:
                way = road[:i+1]
                way_list.append(way)
    #print(way_list)
    distance = []
    for i in range(len(way_list)):
        distance.append(distance_line_road(ws, way_list[i], pos0, ref))
    optimal_way = way_list[distance.index(min(distance))]
    return optimal_way

def change_angle(angle):
    # change angle (radian) to [-pi, pi]
    return np.arctan2(np.sin(angle), np.cos(angle))

def pcontrol(angle, pos0, goal_pose):
    # simple p controller
    kpose = 0.5
    kangle = 0.4
    vmax = 0.1
    wmax = 0.2
    ux = kpose * (goal_pose[0] - pos0[0])
    uy = kpose * (goal_pose[1] - pos0[1])
    theta_r = np.arctan2(uy,ux)
    v = np.sqrt(ux**2 + uy**2)
    w = kangle * (theta_r - change_angle(angle))
    v = max(-vmax,min(v,vmax))
    w = max(-wmax,min(w,wmax))
    return v,w


def check_common_line(poly1, poly2):
    # check if 2 polytopes have a common line (not edge, a LINE)
    epsilon = 1e-5
    (A1, B1)= vertices2halfspace(poly1)
    found = False
    i = len(poly2)
    while i>0 and not found:
        i = i -1
        for j in range(len(B1)):
            if abs(A1[j][0] * poly2[i][0] + A1[j][1] * poly2[i][1] - B1[j]) < epsilon:
                if abs(A1[j][0] * poly2[i-1][0] + A1[j][1] * poly2[i-1][1] - B1[j]) < epsilon:
                    found = True
    return found




def add_vertices_to_polytope(poly1, points):
    epsilon = 1e-5
    P1 = poly1[:]
    for j in range(len(points)):
        coincidence = False
        for vertice in poly1:
            if distance(points[j], vertice) < epsilon:
                coincidence = True
                break
        if not coincidence:
            i = len(P1)
            found_edge = False
            while i>0 and not found_edge:
                i = i-1
                if check_point_in_between(points[j], P1[i], P1[i-1]):
                    found_edge = True
            if found_edge:
                point_list = [x for x in points[j]]
                P1.insert(i, point_list)
    return P1

def correct_vertices(poly1, poly2):
    # provide the common vertices to the representation, even if collinear
    #edge = common_edge(poly1, poly2)
    #print(edge)
    # if edge:
    #     P1 = add_vertices_to_polytope(poly1, edge)
    #     P2 = add_vertices_to_polytope(poly2, edge)
    #     return P1, P2
    #have_common_line = check_common_line(poly1, poly2)
    #if have_common_line:
    P1 = add_vertices_to_polytope(poly1, poly2)
    P2 = add_vertices_to_polytope(poly2, poly1)
    return P1, P2
   # else:
        #return poly1, poly2


def correct_workspace(ws, typical=True):
    # typical: there exists 1 common vertice of the sharing edge and missing only 1
    # non typical: there may be no common vertices but a sharing edge
    ws_c = ws[:]
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i<j:
                if typical:
                    if len(shared_vertices(ws[i], ws[j]))==1:
                        if check_common_line(ws[i], ws[j]):
                            (P1, P2) = correct_vertices(ws_c[i], ws_c[j])
                            ws_c[i] = P1
                            ws_c[j] = P2
                else:
                    if len(shared_vertices(ws[i], ws[j]))<2:
                        if check_common_line(ws[i], ws[j]):
                            (P1, P2) = correct_vertices(ws_c[i], ws_c[j])
                            ws_c[i] = P1
                            ws_c[j] = P2
    return ws_c

def polytopemap_from_gridmap(data_origin,epsilon_outer=0.5,epsilon_inner=0.5,offset_distance=1):
    data = np.copy(data_origin)
    # now transform it into binary image, 0 is black = obstacles + unexplored
    data[data==0] = 255
    data[data==100] = 0
    data[data==-1] = 0
    # cv requires grayscale (8UC1) format
    data_for_cv = np.array(data, dtype=np.uint8)
    """
            First, find the outer boundary of the freespace, by using option RETR_EXTERNAL, chosing only one polygon
            with maximum area as the outer polygon
    """
    contours, hierarchy = cv.findContours(data_for_cv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # countours are given in a strange format [[[[x1, y1]],[[x2, y2]]], ...]
    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        # print('Area of contour:', area)
        if (area >= 1) and (area > max_area): # to remove noisy features (1-2 pixels)
            max_area = area
            # need to use np.flip to save data in [Height, Width] pixel
            outer_polygon = np.array([np.flip(contour[i][0],0) for i in range(len(contour))], dtype=float)
            simplified_outer_polygon = rdp(outer_polygon, epsilon=epsilon_outer)

    """
            Second, find all the inner obstacles by using RETR_LIST
    """

    contours, hierarchy = cv.findContours(data_for_cv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # countours are given in a strange format [[[[x1, y1]],[[x2, y2]]], ...]
    obstacles = []
    for contour in contours:
        area = cv.contourArea(contour)
        # print('Area of contour:', area)
        if (area >= 1) and (area != max_area): # to remove noisy features (1-2 pixels)
            obstacle = np.array([np.flip(contour[i][0],0) for i in range(len(contour))], dtype=float)
            obstacles.append(rdp(obstacle, epsilon = epsilon_inner))

    """
            Third, make polygon, apply offset and decompose
    """

    # enlarge all the obstacles by a safety offset including the boundary
    offset_distance = offset_distance# in cell
    obstacle_list = []
    for obstacle in obstacles:
        obstacle_polygon = gdspy.Polygon(obstacle)
        offset_operation = gdspy.offset(obstacle_polygon, offset_distance)
        offset_result = offset_operation.polygons[0]
        obstacle_list.append(offset_result)

    outer_polygon = gdspy.Polygon(simplified_outer_polygon)
    outer_polygon = gdspy.offset(outer_polygon, -offset_distance)
    free_polygon = gdspy.boolean(outer_polygon, obstacle_list, "not")
    work_space = []

    for free_poly in free_polygon.polygons:
        decompose_result = pd.polygonQuickDecomp(free_poly)
        work_space += decompose_result
    return work_space,simplified_outer_polygon,obstacles

def find_common_vertices(p1, p2):
    ver_list = np.empty((0, 2), float)
    check_list = gdspy.inside(p1, gdspy.Polygon(p2))
    for i in range(len(check_list)):
        if check_list[i]:
            ver_list = np.append(ver_list, np.array([p1[i]]), axis=0)
    return ver_list

# Check intersect with obstacle
def is_intersecting_obstacle(start, end, obstacle_polygon):
    path_line = LineString([start, end])
    for obs in obstacle_polygon:
        if path_line.intersects(obs):
            return True
    return False

def check_consecutive_polytopes(p1, p2):
    check = False
    s = 0
    check_list = gdspy.inside(p1, gdspy.Polygon(p2))
    for i in range(len(check_list)):
        if check_list[i]:
            s = s + 1
    if s == 2:
        check = True
    return check

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

def find_polyp(workspace, points):
    polyp_index = -1
    for i in range(len(workspace)):
        if (gdspy.inside([points], gdspy.Polygon(workspace[i])))[0]:
            polyp_index = i
    return polyp_index

def center(polyp):
    xc = np.matrix(polyp)
    return xc.mean(0)

def sort_vertices_clockwise(vertices):
    # Compute center of polytope
    center = np.mean(vertices, axis=0)

    # Compute angle according to the center
    angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])

    # Sorting the angles in decreasing order
    sort_order = np.argsort(-angles)  # "-" denotes decreasing
    sorted_vertices = vertices[sort_order]

    return sorted_vertices