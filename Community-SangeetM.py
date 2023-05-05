#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:55:37 2022

@author: kurup
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
def get_valid_eig_ind(eig_val):
    i = 0
    eps=0.00001
    sorted_eig_val=np.sort(eig_val)
    for j in range(len(sorted_eig_val)):
        if sorted_eig_val[j]>=eps:
            i=j
            break
    if i==-1:
        return -1
    return np.argsort(eig_val)[i]


def make_adj_mat(edges):
    dim=edges.max()
    adj_mat=np.zeros((dim+1,dim+1))
    for i in edges:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1 
    return adj_mat    

def spec_d_comp_sub(edges):
       
    adj_mat=make_adj_mat(edges)
    D_mat = np.diag(adj_mat.sum(axis=0))
    L_mat = D_mat-adj_mat
    eig_val, eig_vec = np.linalg.eig(L_mat)
    F_vec = eig_vec[get_valid_eig_ind(eig_val)]
    splits_comm = -np.ones(len(adj_mat))
    nodes = np.unique(edges.reshape(-1))
    if(np.all(F_vec>=0) or np.all(F_vec<0)):
        comm_ind=edges.min()
        splits_comm=np.full(len(splits_comm),comm_ind)
        
    else:
        comm_1_ind = np.where(F_vec>=0)[0].min() # index to eigen values greater than or equal to zero
        comm_m0_ind = np.where(F_vec<0)[0].min() # index to eigen values less than zero
        splits_comm=np.full(len(splits_comm),comm_m0_ind)
        splits_comm[np.intersect1d(np.where(F_vec<0)[0],nodes)] = comm_1_ind
    splits_comm = np.vstack([np.arange(len(adj_mat)), splits_comm]).T
    return F_vec, adj_mat, splits_comm


def go_forward(partition):
    vals=partition[:,-1]
    return any(vals==-1)
    
def reverse_dict(dic):
    return dict(zip(dic.values(), dic.keys()))

def get_parts(F_vec):
    return np.where(F_vec>=0)[0],np.where(F_vec<0)[0]

def get_parts_edges(rev_map,comm_1,comm_m0,edges,nodes_rev):
    comm_1_inds = np.array([rev_map[i] for i in nodes_rev[comm_1]])
    map_next_1 = dict(zip(comm_1_inds, np.arange(len(comm_1))))
    comm_m0_inds = np.array([rev_map[i] for i in nodes_rev[comm_m0]])
    map_next_m0 = dict(zip(comm_m0_inds, np.arange(len(comm_m0))))    
    edge_comm_m0 = []
    edge_comm_1 = []
    for i in edges:
        if(i[0] in comm_m0 and i[1] in comm_m0):
            edge_comm_m0.append([map_next_m0[rev_map[i[0]]],map_next_m0[rev_map[i[1]]]])
        if(i[0] in comm_1 and i[1] in comm_1):
            edge_comm_1.append([map_next_1[rev_map[i[0]]],map_next_1[rev_map[i[1]]]])
    return edge_comm_1,edge_comm_m0,map_next_1,map_next_m0


def if_one_partition(comm,rev_map,nodes_rev,partition):
    comm_ind = np.array([rev_map[i] for i in nodes_rev[comm]])
    partition[comm_ind,1]=comm_ind.min()
    return partition

def d_comp_spec(edges, partition, glob_loc_map):
    if(go_forward(partition)):
        rev_map = reverse_dict(glob_loc_map)
        F_vec= spec_d_comp_sub(edges)[0]
        nodes_rev = np.array(list(glob_loc_map.values()))
        comm_1,comm_m0=get_parts(F_vec)
        
        if(len(comm_m0)==0 or len(comm_1)==0):
            if(len(comm_m0)==0):
                partition=if_one_partition(comm_1,rev_map,nodes_rev,partition)
            else:
                partition=if_one_partition(comm_m0,rev_map,nodes_rev,partition)
            
        else:
            edge_comm_1,edge_comm_m0,map_next_1,map_next_m0=get_parts_edges(rev_map, comm_1, comm_m0, edges, nodes_rev)
            if(len(edge_comm_m0)):
                d_comp_spec(np.array(edge_comm_m0),partition, map_next_m0)
            if(len(edge_comm_1)):
                d_comp_spec(np.array(edge_comm_1),partition, map_next_1)
    return partition


def get_adj_v(edges):
    V=[]
    sub_edge=[]
    for i in edges:
        if type(i)==str:
            v1,v2=i.split(" ")
        v1=int(v1)
        v2=int(v2)
        if v1 not in V:
            V.append(v1)
        if v2 not in V:
            V.append(v2)
        sub_edge.append([v1,v2])
    return sub_edge

def get_fb():
    #path="facebook_combined.txt"
    path="facebook_combined.txt"

    f = open(path, "r")
    data=f.read()
    data.splitlines()
    edges=data.splitlines()
    f.close()
    return get_adj_v(edges)
def get_bt():
    pathbit="soc-sign-bitcoinotc.csv"
    data_bit=pd.read_csv(pathbit,names=["a","b","c","d"])
    aa=data_bit[["a","b"]]
    return aa.values


def near_vert(vert,adj_mat):
    vert=adj_mat[vert]
    indices = [index for index, element in enumerate(vert) if element == 1]
    return indices

def assi_unassi(i,parts,adj):
    nei_comm=parts[near_vert(i,adj)].T[1]
    if (len(nei_comm)==0) or (len(nei_comm)==1 and np.all(nei_comm==-1)):
        return -1
    a,b=np.unique(nei_comm,return_counts=True)
    fie=np.argsort(b)[-2:]
    if(len(fie)==1):
      return a[fie]

    elif a[fie[1]]!=-1:
        return a[fie[1]]
    else:
        return a[fie[0]]


###############################################################################
#                    Spectral Decomposition
# RUN THE PROGRAM WITH ONLY ONE OF THE 2 FOLLOWING LINES UNCOMMENTED

#edges=np.array(get_fb()) #uncomment to load fb data
edges=get_bt() # uncomment to load bitcoin data
###############################################################################


dim_ed=edges.max()+1
partition = np.vstack([np.arange(dim_ed),-np.ones(dim_ed)]).T
glob_loc_map = dict(zip(np.arange(dim_ed), np.arange(dim_ed)))
partition = d_comp_spec(edges, partition, glob_loc_map)
adj_mat = make_adj_mat(edges)


un_assi=np.where(partition[:,1]==-1)[0]
for i in un_assi :
    partition[i,1] = assi_unassi(i,partition,adj_mat) #assign for unassigned nodes to its most popular closest node

def disp_plot(edges,parts):
    G = nx.from_edgelist(edges)
    pos = nx.nx_agraph.graphviz_layout(G)
    options = {"node_size": 10, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, node_color=parts[:,1][G.nodes()], cmap = mpl.cm.get_cmap('Spectral'), **options)
    nx.draw_networkx_edges(G, pos, alpha=0.5,width=1)
    plt.tight_layout()
    plt.axis("off")
    plt.show()

disp_plot(edges,partition)




############################################################################################################################################################################################################################################################################################################################
#                                                               Louvain Method

def get_adj_v(edges):
    V=[]
    sub_edge=[]
    for i in edges:
        if type(i)==str:
            v1,v2=i.split(" ")
        else:
            v1,v2=i
        v1=int(v1)
        v2=int(v2)
        if v1 not in V:
            V.append(v1)
        if v2 not in V:
            V.append(v2)
        sub_edge.append([v1,v2])
    adj_lov=[[0 for i in range(max(V)+1)] for j in range(max(V)+1)]
    for i in sub_edge:
        v1,v2=i
        adj_lov[v1][v2]=1
        adj_lov[v2][v1]=1
    #return sub_edge,V.sort()
    return adj_lov,sorted(V),np.array(sub_edge)

def get_fb():
    pathf="facebook_combined.txt"
    f = open(pathf, "r")
    data=f.read()
    data.splitlines()
   
    edges= np.array([[int(j) for j in i.split()] for i in data.splitlines()])
    if edges.min()>0:
        edges=edges-edges.min()
    f.close()
    return get_adj_v(edges)


def get_bt():
    pathbit="soc-sign-bitcoinotc.csv"
    data_bit=pd.read_csv(pathbit,names=["a","b","c","d"])
    aa=data_bit[["a","b"]]
    return get_adj_v(aa.values-1)




def get_comm_index(vert):
    for i in range(len(communities)):
        if vert in communities[i]:
            return i
    return -1

def near_vert(vert):
    vert=adj_lov[vert]
    indices = [index for index, element in enumerate(vert) if element == 1]
    return indices

def degree_vert(vert):
    return sum(adj_lov[vert])
df=[]
###############################################################################
#UNCOMMENT ONLY ONE OF THE FOLLOWING 2 LINES AT A TIME

adj_lov,V,edges=get_fb() #UNCOMMENT TO GET FB DATA
#adj_lov,V,edges=get_bt() #UNCOMMENT TO GET BITCOIN DATA
###############################################################################

m=len(edges)
communities=[[i] for i in V]
clus_no={V[i]:i for i in range(len(V))}
clus_verts={i:[i] for i in range(len(V))}
for i in V:
    df.append([i,get_comm_index(i),near_vert(i),degree_vert(i)])
graph=pd.DataFrame(df,columns=["Vertex","Community","Neighbours","Degree"])

comm_ind={i:i for i in V}
comm_list={i:[i] for i in V}
###############################################################################


def get_comm_index(vert):
    return comm_ind[vert]
def near_vert(vert):
    return graph[graph["Vertex"]==vert]["Neighbours"].values[0]

def degree_vert(vert):
    return graph[graph["Vertex"]==vert]["Degree"].values[0]

def mod_calc(vertices):
    sig_in=0
    deg_sum=0
    for i in vertices:
        deg_sum+=degree_vert(i)
        for j in vertices:
            if adj_lov[i][j]==1:
                sig_in+=1
    sig_in=sig_in/(2*m)
    deg_sum=deg_sum/(2*m)
    deg_sum*=deg_sum
    return sig_in-deg_sum

def iter_once():    
    jump=[]    
    for i in V: 
        temp=0   
        neigh=near_vert(i)
        
        for nei in neigh:
            new_clus_no=get_comm_index(nei)
            pre_clus_no=get_comm_index(i)            
            if new_clus_no!=pre_clus_no:
                new_clus=comm_list[new_clus_no].copy()
                pre_clus=comm_list[pre_clus_no].copy()
                
                old_Q=mod_calc(new_clus)+mod_calc(pre_clus)
                new_clus.append(i)
                pre_clus.remove(i)
                new_Q=mod_calc(new_clus)+mod_calc(pre_clus)
                delta=new_Q-old_Q
                #print(f"i: {i}     nei: {nei}   delta: {delta}")
                if delta>temp:
                    temp=delta
                    jump=[i,nei,pre_clus_no,new_clus_no]
            else:
                #print(f"i: {i}     nei: {nei}  No need to compute")
                pass
    return jump

def lovi_run():    
    jump=[1]
    while (not jump==[]):
        jump=iter_once()
        if not jump==[]:
            vert1=jump[0]
            vert2=jump[1]
            comm_list[comm_ind[vert1]].remove(vert1)
            if comm_list[comm_ind[vert1]]==[]:
                del comm_list[comm_ind[vert1]]
            comm_ind[vert1]=comm_ind[vert2]
            comm_list[comm_ind[vert1]].append(vert1)
    comms=np.vstack([list(comm_ind.keys()),list(comm_ind.values())]).T
    print(f"Communities total :{len(np.unique(comms.T[1]))}")
    return comms

communities=lovi_run()

disp_plot(edges,communities)


