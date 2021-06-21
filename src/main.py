#!/usr/bin/env python
# coding: utf-8

# In[4]:


from time import time
import os

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches

# for RRT
from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace


# In[5]:


# plot a 2D projection map of the given 3D map

def plot2D(env, axis=(0,1),show=True, figsize=None, startgoal=True):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    if startgoal:
        if "s" in env.keys():
            sx,sy = env["s"][:2]
            ax.scatter(sx,sy,color="red", label="start")
        if "t" in env.keys():
            gx,gy = env["t"][:2]
            ax.scatter(sx,sy,color="red", label="goal")

    if "o" in env.keys():
        for xyz in env["o"]:
            i,j = axis
            x,y,w,h = xyz[i], xyz[j], xyz[i+3]-xyz[i], xyz[j+3]-xyz[j]
            r = patches.Rectangle(xy=(x, y), width=w, height=h, color="green")
            ax.add_patch(r)
            plt.axis('scaled')
            ax.set_aspect('equal')
    if "b" in env.keys():
        for xyz in env["b"]:
            x,y,w,h = xyz[i], xyz[j], xyz[i+3]-xyz[i], xyz[j+3]-xyz[j]
            r = patches.Rectangle(xy=(x, y), width=w, height=h, color="black", fill=False)
            ax.add_patch(r)
            plt.axis('scaled')
            ax.set_aspect('equal')
    if show: plt.show()


# In[6]:


# compute coordinate of 6 faces of the given box

def calc_plane(x1,y1,z1,x2,y2,z2):
    base = [x1,y1,z1,x2,y2,z2]
    out = []
    for i in range(6):
        res = base[:]
        res[i] = res[(i+3)%6]
        out.append(res)
    return out


# In[7]:


# check if the given line segment and plane intersects
def is_cross_with_plane(linseg, plane):
    for i in range(3):
        if plane[i] == plane[i+3]:break
    if linseg[i] >  linseg[i+3]:  linseg[:3], linseg[3:] = linseg[3:],linseg[:3]
    #print(linseg, linseg[i], linseg[i+3], plane[i])
    if not (linseg[i] <= plane[i] <= linseg[i+3]): return False,[-1,-1,-1]
    if linseg[i]==linseg[i+3]: return False,[-1,-1,-1]
    ratio = (plane[i]-linseg[i])/(linseg[i+3] - linseg[i])
    res = [0,0,0]
    for j in range(3):
        res[j] = linseg[j] + ratio*(linseg[j+3]-linseg[j])
        if i != j and not (plane[j] <= res[j] <= plane[j+3] or plane[j+3] <= res[j] <= plane[j]): return False,[-1,-1,-1]
    return True,res


# In[8]:


# collision checker (input: line segment, box)

def is_cross_with_rect(linseg, rectangle):
    planes = calc_plane(*rectangle)
    for plane in planes:
        res = is_cross_with_plane(linseg, plane)
        if res[0]: return res
    return res


# In[9]:


# check if the given point is inside boundary

def is_inside(pos, boundary):
    return boundary[0]<=pos[0]<=boundary[3] and boundary[1]<=pos[1]<=boundary[4] and boundary[2]<=pos[2]<=boundary[5]


# In[10]:


# utility function transforms from/to coordinate, discretized coordinate, or 1d index

def pos2idx(x,y,z, Y, Z):
    return x*(Y*Z) + y*(Z) + z

def idx2pos(idx, Y, Z):
    x,idx = divmod(idx, Y*Z)
    y,z = divmod(idx, Z)
    return x,y,z

def idx2pos(idx, Y, Z):
    x,idx = divmod(idx, Y*Z)
    y,z = divmod(idx, Z)
    return x,y,z

def xyz2idx(x,y,z):
    return [int((w-wmin)*resol) for w,wmin in zip((x,y,z),(xmin,ymin,zmin))]

def idx2xyz(i,j,k):
    return [wmin+w/resol for w,wmin in zip((i,j,k),(xmin,ymin,zmin))]

# L2 distance function
def dist_L2(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

# L2 distance with pre-compute
dist_dict = {(i+1):(i+1)**0.5 for i in range(3)}
def dist(x1,y1,z1,x2,y2,z2):
    return dist_dict[abs(x1-x2)+abs(y1-y2)+abs(z1-z2)]


# In[11]:


# Dijkstra algorithm on grid

# ゴールに到達したら離脱

def dijkstra(sx,sy,sz,gx,gy,gz):

    from heapq import heappush,heappop
    INF = 10**5
    si,sj,sk = xyz2idx(sx,sy,sz)
    gi,gj,gk = xyz2idx(gx,gy,gz)
    q = [(0,pos2idx(si,sj,sk,Y,Z))]
    D = [[[INF]*Z for _ in range(Y)] for _ in range(X)]
    prev = [[[-1]*Z for _ in range(Y)] for _ in range(X)]
    D[si][sj][sk] = 0
    d = [-1,0,1]

    while q:
        d_, idx1 = heappop(q)
        i1,j1,k1 = idx2pos(idx1,Y,Z)
        if d_ > D[i1][j1][k1]: continue
        x1,y1,z1 = idx2xyz(i1,j1,k1)
        for di in d:
            for dj in d:
                for dk in d:

                    if di==dj==dk==0: continue
                    i2,j2,k2 = i1+di, j1+dj, k1+dk
                    x2,y2,z2 = idx2xyz(i2,j2,k2)
                    if not is_inside([x2,y2,z2],env["b"][0]): continue
                    idx2 = pos2idx(i2,j2,k2,Y,Z)

                    cost = dist(i1,j1,k1,i2,j2,k2) / resol
                    if D[i2][j2][k2] > D[i1][j1][k1] + cost:

                        # collision check
                        flag = 0
                        for rect in env["o"]:
                            flag, _ = is_cross_with_rect([x1,y1,z1,x2,y2,z2], rect[:-3])
                            if flag: break
                        if flag: continue
                        
                        D[i2][j2][k2] = D[i1][j1][k1] + cost
                        prev[i2][j2][k2] = (i1,j1,k1)
                        if (i2,j2,k2) == (gi,gj,gk): return D, prev
                        heappush(q, (D[i2][j2][k2], idx2))

    return D, prev


# In[12]:


# Weighted A*

def a_star(sx,sy,sz,gx,gy,gz,h,eps=1):

    from heapq import heappush,heappop
    INF = 10**5
    si,sj,sk = xyz2idx(sx,sy,sz)
    gi,gj,gk = xyz2idx(gx,gy,gz)
    q = [(h(sx,sy,sz),pos2idx(si,sj,sk,Y,Z))]
    D = [[[INF]*Z for _ in range(Y)] for _ in range(X)]
    prev = [[[-1]*Z for _ in range(Y)] for _ in range(X)]
    D[si][sj][sk] = 0
    d = [-1,0,1]

    while q:
        d_, idx1 = heappop(q)
        i1,j1,k1 = idx2pos(idx1,Y,Z)
        if (i1,j1,k1) == (gi, gj, gk): return D, prev
        x1,y1,z1 = idx2xyz(i1,j1,k1)
        for di in d:
            for dj in d:
                for dk in d:

                    if di==dj==dk==0: continue
                    i2,j2,k2 = i1+di, j1+dj, k1+dk
                    x2,y2,z2 = idx2xyz(i2,j2,k2)
                    if not is_inside([x2,y2,z2],env["b"][0]): continue
                    idx2 = pos2idx(i2,j2,k2,Y,Z)

                    cost = dist(i1,j1,k1,i2,j2,k2) / resol
                    h_val = h(x2,y2,z2)*eps
                    if D[i2][j2][k2] > D[i1][j1][k1] + cost + h_val:

                        flag = 0
                        for rect in env["o"]:
                            flag, _ = is_cross_with_rect([x1,y1,z1,x2,y2,z2], rect[:-3])
                            if flag: break
                        if flag: continue
                        
                        D[i2][j2][k2] = D[i1][j1][k1] + cost
                        heappush(q, (D[i2][j2][k2]+h_val, idx2))
                        prev[i2][j2][k2] = (i1,j1,k1)

    return D, prev


# In[13]:


# Dijkstra on visiblity graph

def dijkstra0(start = 0, INF=10**20):
    from heapq import heappop, heappush
    d = [INF for i in range(len(G))]
    d[start] = 0
    que = []
    heappush(que, (0, start))
    prev = [-1 for i in range(len(G))]
    while que:
        d_, v = heappop(que)
        if d[v] < d_: continue
        for u, c in G[v].items():
            if d[u] > d[v] + c:
                d[u] = d[v] + c
                heappush(que, (d[u], u))
                prev[u] = v
    return d, prev


# In[ ]:


if __name__ == "__main__":
    ### creating a dictionary which contains maps ###
    print("Loading Maps...")
    path0 = "maps"

    files = os.listdir(path0)

    env_dict = {}

    for f in files:
        path = "/".join([path0, f])

        with open(path) as F:
            L = F.readlines()
            *L, = map(str.strip, L)

        env = {}
        mode = 0
        for line in L:
            if "start" in line or "Start" in line:
                mode = 1
                continue
            elif "end" in line or "End" in line or "stop" in line or "goal" in line:
                mode = 2
                continue
            if mode==1:
                for t in "(,)[];": line = line.replace(t, " ")
                line = line.split()
                res = []
                for s in line:
                    try: s = float(s)
                    except: continue
                    res.append(s)
                env["s"] = res
                mode = 0
                continue
            elif mode==2:
                for t in "(,)[];": line = line.replace(t, " ")
                line = line.split()
                res = []
                for s in line:
                    try: s = float(s)
                    except: continue
                    res.append(s)
                env["t"] = res
                mode = 0
                continue

            if not line: continue
            line = line.split()
            if line[0] == "boundary":
                *b, = map(float, line[1:])
                if "b" in env.keys(): env["b"].append(b)
                else: env["b"] = [b]
            elif line[0] == "block":
                *o, = map(float, line[1:])
                if "o" in env.keys(): env["o"].append(o)
                else: env["o"] = [o]
            else:
                continue

            env_dict[f[:-4]] = env
            
            
    ### Grid + Dijkstra ###
    print("\n")
    print("Grid + Dijkstra\n")
    for env_name, env in env_dict.items():
        print(f"Environment: {env_name}")
        time_start = time()

        sx,sy,sz = 0,0,0
        gx,gy,gz = 5,5,0
        if env_name=="single_cube":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 8,8,3
        elif env_name=="window":
            sx,sy,sz = 5,-3,3
            gx,gy,gz = 5, 19.5, 5
        elif env_name == "maze":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 0,-12,3

        resol = 5
        xmin,ymin,zmin,xmax,ymax,zmax, *_ = env["b"][0]
        xmin,ymin,zmin = map(lambda x:(2*int(x)-1)//2, (xmin, ymin, zmin))
        xmax,ymax,zmax = map(lambda x:(2*int(x)+2)//2, (xmax,ymax,zmax))
        boundary = [xmin, ymin, zmin, xmax, ymax, zmax]

        X,Y,Z = (xmax-xmin)*resol+1, (ymax-ymin)*resol+1,(zmax-zmin)*resol+1

        D = [[[0]*Z for _ in range(Y)] for _ in range(X)]
        (sx,sy,sz) = env["s"] if "s" in env.keys() else (sx,sy,sz)
        si,sj,sk = xyz2idx(sx,sy,sz)
        (gx,gy,gz) = env["t"] if "t" in env.keys() else (gx,gy,gz)
        gi,gj,gk = xyz2idx(gx,gy,gz)

        D, prev = dijkstra(sx,sy,sz,gx,gy,gz)

        path = [(gi,gj,gk)]
        i,j,k = gi,gj,gk
        while True:
            if prev[i][j][k]==-1: break
            ni,nj,nk = prev[i][j][k]
            path.append((ni,nj,nk))
            i,j,k = ni,nj,nk

        path = path[::-1]

        print(f"Optimal cost: {D[gi][gj][gk]}")
        print(f"Computing time: {time()-time_start}")
        n = 0
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if D[i][j][k] < 10**5: n += 1
        print(f"Number of considered nodes: {n}")
        print("Trajectory: ")
        # to visualize, roof of the room is removed
        if env_name == "room":
            env = {k:v[:] for k,v in env.items()}
            tmp = env["o"].pop()
        for d1,d2 in [(0,1),(1,2),(2,0)]:
            plot2D(env, (d1,d2), show=False,figsize=(5,5),startgoal=False)
            for idx in range(len(path)-1):
                i,j,k = path[idx]
                i2,j2,k2 = path[idx+1]
                x,y,z = idx2xyz(i,j,k)
                x2,y2,z2 = idx2xyz(i2,j2,k2)
                xx,xx2 = (x,y,z)[d1],(x2,y2,z2)[d1]
                yy,yy2 = (x,y,z)[d2],(x2,y2,z2)[d2]
                plt.plot([xx,xx2],[yy,yy2],color="red")
            plt.show()
            
    
    ### Grid + weighted A*###
    print("\n")
    print("Grid + Weighted A*\n")
    # weight parameter (can be changed)
    eps=1

    for env_name, env in env_dict.items():
        print(f"Environment: {env_name}")
        time_start = time()

        sx,sy,sz = 0,0,0
        gx,gy,gz = 5,5,0
        if env_name=="single_cube":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 8,8,3
        elif env_name=="window":
            sx,sy,sz = 5,-3,3
            gx,gy,gz = 5, 19.5, 5
        elif env_name == "maze":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 0,-12,3

        resol = 5
        xmin,ymin,zmin,xmax,ymax,zmax, *_ = env["b"][0]
        xmin,ymin,zmin = map(lambda x:(2*int(x)-1)//2, (xmin, ymin, zmin))
        xmax,ymax,zmax = map(lambda x:(2*int(x)+2)//2, (xmax,ymax,zmax))
        boundary = [xmin, ymin, zmin, xmax, ymax, zmax]

        X,Y,Z = (xmax-xmin)*resol+1, (ymax-ymin)*resol+1,(zmax-zmin)*resol+1

        D = [[[0]*Z for _ in range(Y)] for _ in range(X)]
        (sx,sy,sz) = env["s"] if "s" in env.keys() else (sx,sy,sz)
        si,sj,sk = xyz2idx(sx,sy,sz)
        (gx,gy,gz) = env["t"] if "t" in env.keys() else (gx,gy,gz)
        gi,gj,gk = xyz2idx(gx,gy,gz)

        D, prev = a_star(sx,sy,sz,gx,gy,gz,h=lambda i,j,k: dist_L2(i,j,k,gx,gy,gz), eps=eps)

        path = [(gi,gj,gk)]
        i,j,k = gi,gj,gk
        while True:
            if prev[i][j][k]==-1: break
            ni,nj,nk = prev[i][j][k]
            path.append((ni,nj,nk))
            i,j,k = ni,nj,nk

        path = path[::-1]

        print(f"Optimal cost: {D[gi][gj][gk]}")
        print(f"Computing time: {time()-time_start}")
        n = 0
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if D[i][j][k] < 10**5: n += 1
        print(f"Number of considered nodes: {n}")
        print("Trajectory: ")
        # to visualize, roof of the room is removed
        if env_name == "room":
            env = {k:v[:] for k,v in env.items()}
            tmp = env["o"].pop()
        for d1,d2 in [(0,1),(1,2),(2,0)]:
            plot2D(env, (d1,d2), show=False,figsize=(5,5),startgoal=False)
            for idx in range(len(path)-1):
                i,j,k = path[idx]
                i2,j2,k2 = path[idx+1]
                x,y,z = idx2xyz(i,j,k)
                x2,y2,z2 = idx2xyz(i2,j2,k2)
                xx,xx2 = (x,y,z)[d1],(x2,y2,z2)[d1]
                yy,yy2 = (x,y,z)[d2],(x2,y2,z2)[d2]
                plt.plot([xx,xx2],[yy,yy2],color="red")
            plt.show()
            
    ### Visibility Graph  + Dijkstra ###
    print("\n")
    print("Visibility Graph + Dijkstra\n")
    
    # resolution parameter
    resol3 = 5
    
    for env_name, env in env_dict.items():
        print(f"Environment: {env_name}")
        start = time()

        sx,sy,sz = 0,0,0
        gx,gy,gz = 5,5,0
        if env_name=="single_cube":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 8,8,3
        elif env_name=="window":
            sx,sy,sz = 5,-3,3
            gx,gy,gz = 5, 19.5, 5
        elif env_name == "maze":
            sx,sy,sz = 0,0,3
            gx,gy,gz = 0,-12,3

        (sx,sy,sz) = env["s"] if "s" in env.keys() else (sx,sy,sz)
        (gx,gy,gz) = env["t"] if "t" in env.keys() else (gx,gy,gz)

        resol = 5
        xmin,ymin,zmin,xmax,ymax,zmax, *_ = env["b"][0]
        xmin,ymin,zmin = map(lambda x:(2*int(x)-1)//2, (xmin, ymin, zmin))
        xmax,ymax,zmax = map(lambda x:(2*int(x)+2)//2, (xmax,ymax,zmax))
        boundary = [xmin, ymin, zmin, xmax, ymax, zmax]

        nodes = []
        
        resol2 = resol
        margin = 1/resol2
        for rect in env["o"]:
            rect2 = [rect[i]-margin if i<3 else rect[i]+margin for i in range(6)]
            for i,j,k in [(0,0,0), (0,1,3), (1,2,3), (2,0,3)]:
                base = rect2[:-3]
                base[i], base[j] = rect2[(i+k)%6], rect2[(j+k)%6]
                for x in range(3):
                    base2 = base[:]
                    mode = 0 if i+j and (x==i or x==j) else 1
                    base2[x] = rect2[x + 3*mode]
                    for y in range(int(abs(base2[x]-base[x]))*resol3+1):
                        node = base[:]
                        if mode: node[x] += y/resol
                        else: node[x] -= y/resol

                        if not is_inside(node, env["b"][0]): continue

                        ok=1
                        for rect in env["o"]:
                            if is_inside(node, rect):
                                ok=0
                                break

                        if not ok: continue

                        nodes.append(node)

        nodes = list(set(tuple(node) for node in nodes))

        N = len(nodes)
        print(f"Number of nodes: {N}")
        G0=[{} for _ in range(N)]
        for i in range(N-1):
            if i and i%100==0: print(f"Constructing edges of {i}-th node...")
            x1,y1,z1 = nodes[i]
            for j in range(i+1,N):
                x2,y2,z2 = nodes[j]
                ok=1
                for rect in env["o"]:
                    if is_cross_with_rect([x1,y1,z1,x2,y2,z2], rect[:-3])[0]:
                        ok=0
                        break
                if ok: G0[i][j] = G0[j][i] = dist_L2(x1,y1,z1,x2,y2,z2)

        G = [g.copy() for g in G0]
        G.append({})
        G.append({})

        for i in range(N):
            x,y,z = nodes[i]

            ok=1
            for rect in env["o"]:
                if is_cross_with_rect([sx,sy,sz,x,y,z], rect[:-3])[0]:
                    ok=0
                    break
            if ok: G[-2][i] = G[i][-2] = dist_L2(sx,sy,sz,x,y,z)

            ok=1
            for rect in env["o"]:
                if is_cross_with_rect([gx,gy,gz,x,y,z], rect[:-3])[0]:
                    ok=0
                    break
            if ok: G[-1][i] = G[i][-1] = dist_L2(gx,gy,gz,x,y,z)

        time1 = time() - start
        print(f"Number of edges: {len([1 for i in range(len(G)) for j in range(len(G)) if i<j])}")
        start = time()

        d2, prev2 = dijkstra0(N)

        path = [(gx,gy,gz)]
        v = prev2[N+1]
        while True:
            if v==N: break
            path.append(nodes[v])
            v = prev2[v]
        path.append((sx,sy,sz))

        path = path[::-1]

        time2 = time() - start
        print(f"Optimal cost: {d2[N+1]}\n")
        print(f"Time to construct graph: {time1}")
        print(f"Time to solve DSP: {time2}")
        print(f"Total time: {time1 + time2}\n")
        print(f"Number of considered nodes: {len([1 for i in range(len(d2)) if d2[i] < 10**20 ])}")
        print("Trajectory: ")
        # to visualize, roof of the room is removed
        if env_name == "room":
            env = {k:v[:] for k,v in env.items()}
            tmp = env["o"].pop()
        for d1,d2 in [(0,1),(1,2),(2,0)]:
            plot2D(env, (d1,d2), show=False,figsize=(5,5),startgoal=False)
            for i in range(len(path)-1):
                (x,y,z),(x2,y2,z2) = path[i],path[i+1]
                xx,xx2 = (x,y,z)[d1],(x2,y2,z2)[d1]
                yy,yy2 = (x,y,z)[d2],(x2,y2,z2)[d2]
                plt.plot([xx,xx2],[yy,yy2],color="red")
            plt.show()
            
            
    ### RRT ###
    print("\n")
    print("RRT\n")
    for env_name, env in env_dict.items():
        print(f"Environment: {env_name}")
        time_start = time()

        x_init = 0,0,0
        x_goal = 5,5,0
        if env_name=="single_cube":
            x_init = 0,0,3
            x_goal = 8,8,3
        elif env_name=="window":
            x_init = 5,-3,3
            x_goal = 5, 19.5, 5
        elif env_name == "maze":
            x_init = 0,0,3
            x_goal = 0,-12,3

        x_init = tuple(env["s"]) if "s" in env.keys() else x_init
        x_goal = tuple(env["t"]) if "t" in env.keys() else x_goal

        *X_dimensions, = zip(env["b"][0][:3], env["b"][0][3:6])
        X_dimensions = np.array(X_dimensions)
        Obstacles = []
        for o in env["o"]:
            Obstacles.append(tuple(o[:6]))
        Obstacles = np.array(Obstacles)

        Q = np.array([(1,1,1)])  # length of tree edges
        r = 0.01  # length of smallest edge to check for intersection with obstacles
        max_samples = 1<<20  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal

        # create search space
        X = SearchSpace(X_dimensions, Obstacles)

        # create rrt_search
        rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
        path = rrt.rrt_search()

        cost = 0
        for i in range(len(path)-1):
            (x,y,z),(x2,y2,z2) = path[i],path[i+1]
            cost += dist_L2(x,y,z,x2,y2,z2)
        print("Optimal cost: ", cost)
        print(f"Computing time: {time()-time_start}")
        print(f"Number of considered nodes: {rrt.trees[0].V_count}")
        print("Trajectory: ")
        # to visualize, roof of the room is removed
        if env_name == "room":
            env = {k:v[:] for k,v in env.items()}
            tmp = env["o"].pop()
        for d1,d2 in [(0,1),(1,2),(2,0)]:
            plot2D(env, (d1,d2), show=False,figsize=(5,5),startgoal=False)
            for i in range(len(path)-1):
                (x,y,z),(x2,y2,z2) = path[i],path[i+1]
                xx,xx2 = (x,y,z)[d1],(x2,y2,z2)[d1]
                yy,yy2 = (x,y,z)[d2],(x2,y2,z2)[d2]
                plt.plot([xx,xx2],[yy,yy2],color="red")
            plt.show()


# In[ ]:




