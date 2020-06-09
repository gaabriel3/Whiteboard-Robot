import numpy as np
import csv
from PIL import Image, ImageDraw, ImageOps
import os
import math
import tsp
import itertools
import time
import matplotlib.pyplot as plt

w_e = 12 #eraser width
l_e = 12 #eraser length
w_b = 250 #board width
l_b = 125 #board height

########################### INITIAL GRID PROCESSING ###########################

# BASELINE GRID
def fill_grid(image):
    width, height = image.size
    pix_val_R = list(image.getdata(0))
    pix_val_G = list(image.getdata(1))
    pix_val_B = list(image.getdata(2))

    grid = []

    #fill grid with 0's
    for i in range(height):
        row = []
        for j in range(width):
            row.append(0)
        grid.append(row)

    #find black spots
    count = 1
    nodes = []
    for i in range(height):
        for j in range(width):
            it = i*width + j
            if (pix_val_R[it] < 100):
                grid[i][j] = 1
                n = [count, i, j]
                count = count + 1
                nodes.append(n)

    return grid

# SMALLER GRID
def compartmentalize_grid(grid):
    img_width = len(grid[0])
    img_length = len(grid)
    L = int(l_b / l_e) #no of lengthwise chunks
    W = int(w_b / w_e) #no of widthwise chunks
    p_L = int(img_length / L) #pixels in each lengthwise chunk
    p_W = int(img_width / W) #pixels in each widthwise chunk
    new_grid = []

    for i in range(L):
        row = []
        for j in range(W):
            row.append(0)
        new_grid.append(row)

    for i in range(L):
        for j in range(W):
            min_x = 2000
            max_x = 0
            min_y = 2000
            max_y = 0
            for a in range(p_L):
                it_l = a + p_L*i
                for b in range(p_W):
                    it_w = b + p_W*j
                    min_x = min(min_x, it_w)
                    max_x = max(max_x, it_w)
                    min_y = min(min_y, it_l)
                    max_y = max(max_y, it_l)
                    if grid[it_l][it_w] == 1:
                        new_grid[i][j] = 1
    return new_grid

def shuffle_grid(grid, image):
    width, height = image.size
    img_width = len(grid[0])
    img_length = len(grid)
    L = round(l_b / l_e) #no of lengthwise chunks
    W = round(w_b / w_e) #no of widthwise chunks
    p_L = round(img_length / L) #pixels in each lengthwise chunk
    p_W = round(img_width / W) #pixels in each widthwise chunk
    best_grid = []
    best_density = 1000000
    best_down = -1
    best_right = -1
    ig = image

    for md_it in range(1, int(p_L/2)):
        md = md_it*2
        for mr_it in range(1, int(p_W/2)):
            mr = mr_it*2
            move_down = md
            move_right = mr
            new_grid = []

            for i in range(L+1):
                row = []
                for j in range(W+1):
                    row.append(0)
                new_grid.append(row)

            for i in range(L+1):
                for j in range(W+1):
                    for a in range(p_L):
                        if (i == 0):
                            if (a >= move_down):
                                break
                            it_l = a
                        else:
                            it_l = a + p_L*(i-1) + move_down

                        for b in range(p_W):
                            if (j == 0):
                                if (b >= move_right):
                                    break
                                it_w = b
                            else:
                                it_w = b + p_W*(j-1) + move_right

                            try:
                                if grid[it_l][it_w] == 1:
                                    new_grid[i][j] = 1
                            except:
                                continue
            d = grid_density(new_grid)
            if (d < best_density):
                best_density = d
                best_grid = new_grid
                best_down = move_down
                best_right = move_right

    result = [best_grid, best_down, best_right]
    return result

def grid_density(grid):
    num_one = 0
    total = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (grid[i][j] == 1):
                num_one += 1
            total += 1
    return num_one / total

# CREATE INITIAL CLUSTERS - STEP 1
def separate_clusters(G):
    C = {}
    first = True

    for i in range(len(G)):
        for j in range(len(G[0])):
            #not 0
            if G[i][j] != 0:
                #check surroundings
                label = 0
                for k in range(i-1, i): #up
                    if k >= 0:
                        num = G[k][j]
                        if (num != 0):
                            label = num
                if label == 0:
                    for k in range(j-1, j): #left
                        if k >= 0:
                            num = G[i][k]
                            if (num != 0):
                                label = num

                #found/didnt find
                coords = [i, j]
                if first == True:
                    label = 1
                    G[i][j] = 1
                    C[label] = coords
                    first = False
                elif label != 0:
                    G[i][j] = label
                    l = C[label]
                    if type(l[0]) == int:
                        new_l = []
                        new_l.append(l[0])
                        new_l.append(l[1])
                        l = []
                        l.append(new_l)
                    l.append(coords)
                    C[label] = l #FIX
                else:
                    m_key = max_key(C)
                    new_label = m_key + 1
                    C[new_label] = list(coords)
                    G[i][j] = new_label
    return [C, G]

# PROCESS CLUSTERS - STEP 2
def clean_clusters(C):
    for key in C:
        coords = C[key]
        if type(coords[0]) == int:
            C[key] = [C[key]]
    return C

# PROCESS CLUSTERS - STEP 3
def combine_clusters(C, threshold):
    first = True
    new_keys = {}

    for key in C:
        if first == False:
            new_key = find_closest_cluster(C, threshold, key)
            if (new_key != 0):
                try:
                    new_keys[new_key].append(key)
                except:
                    new_keys[new_key] = []
                    new_keys[new_key].append(key)
        first = False

    #alter C
    new_key_sets = combine_new_keys(new_keys)
    for i in range(len(new_key_sets)):
        cur_set = new_key_sets[i]
        orig = cur_set[0]
        for n in range(1, len(cur_set)):
            next_key = cur_set[n]
            next_coords = C[next_key]
            for j in range(len(next_coords)):
                C[orig].append(next_coords[j])
            del C[next_key]

    #fix C numbers
    i = 1
    new_C = {}
    for key in C:
        new_C[i] = C[key]
        i = i + 1

    return new_C

# CREATE GRID WITH LABELED CLUSTERS
def create_nummed_grid(C, G):
    for key in C:
        coords = C[key]
        for i in range(len(coords)):
            cur_coord = coords[i]
            x = cur_coord[0]
            y = cur_coord[1]
            G[x][y] = key
    return G

# FIND THE DENSITY OF EACH CLUSTER
def density_of_clusters(img, C):
    img_width, img_length = img.size
    L = int(l_b / l_e) #no of lengthwise chunks
    W = int(w_b / w_e) #no of widthwise chunks
    p_L = int(img_length / L) #pixels in each lengthwise chunk
    p_W = int(img_width / W) #pixels in each widthwise chunk
    densities = {}
    pixels = img.load()

    for cl in C:
        coords = C[cl]
        densities[cl] = 0
        total_pixels = 0
        d = 0
        for i in range(len(coords)):
            pos = coords[i]
            x = pos[0]
            y = pos[1]

            for a in range(p_L):
                it_l = a + p_L*x
                for b in range(p_W):
                    it_w = b + p_W*y
                    if (it_w >= 0 and it_w < img_width and it_l >= 0 and it_l < img_length):
                        if (pixels[it_w, it_l][0] == 0):
                            d = d + 1
                        total_pixels = total_pixels + 1
        densities[cl] = d / total_pixels

    return densities

# ALL HELPER FUNCTIONS

def max_key(d):
    m = -1
    for key in d:
        if key > m:
            m = key
    return int(m)

def combine_new_keys(new_keys):
    all_sets = []
    for key in new_keys:
        orig = key
        new = new_keys[key]
        cur_set = []
        f = -1
        #1 check if there is a set for orig
        for i in range(len(all_sets)): #access old set
            s = all_sets[i]
            for j in range(len(s)):
                if orig == s[j]:
                    cur_set = s
                    f = i
                    break
        if f == -1: #create new set
            cur_set = [orig]
        #2 add all new keys to that set
        for i in range(len(new)):
            cur_set.append(new[i])
        #3 save set to array
        if f == -1:
            all_sets.append(cur_set)
        else:
            all_sets[f] = cur_set
    return all_sets

def find_closest_cluster(C, threshold, key):
    initial_coords = C[key]
    for old_key in C:
        if old_key < key:
            old_coords = C[old_key]
            for i in range(len(old_coords)):
                old_test_coords = old_coords[i]
                x2 = old_test_coords[0]
                y2 = old_test_coords[1]
                for j in range(len(initial_coords)):
                    cur_test_coords = initial_coords[j]
                    x1 = cur_test_coords[0]
                    y1 = cur_test_coords[1]
                    dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
                    if dist <= threshold:
                        return old_key
    return 0

###############################################################################
###############################################################################
################### ORDERING CLUSTERS - MINIMIZING DISTANCE ###################

# CREATE COST MATRIX AND ASSOCIATED VERTEX MATRIX
def create_weighted_graph(c_list):
    cost_matrix = []
    vertex_matrix = []
    num_c = len(c_list)

    for outer_key in c_list:
        cost_row = []
        vertex_row = []
        outer_coords = c_list[outer_key]
        for inner_key in c_list:
            inner_coords = c_list[inner_key]
            if (outer_key == inner_key):
                cost_row.append(0)
                vertex_row.append([0,0,0,0])
            else:
                D = min_dist(outer_coords, inner_coords)
                d = D[0]
                v = D[1]
                cost_row.append(d)
                vertex_row.append(v)
        cost_matrix.append(cost_row)
        vertex_matrix.append(vertex_row)

    return [cost_matrix, vertex_matrix]

# RUN TSP ON CLUSTER LIST
def TSP_clusters(cost_matrix):
    r = range(len(cost_matrix))
    # Dictionary of distance
    dist = {(i, j): cost_matrix[i][j] for i in r for j in r}
    l = tsp.tsp(r, dist)
    first = l[1][0]
    last = l[1][-1]
    d = l[0] - cost_matrix[last][first]
    for i in range(len(l[1])):
        l[1][i] = l[1][i] + 1
    ret = [l[1], d] #path, distance
    return ret

# GIVE THE ORDER OF THE VERTICES IN BETWEEN CLUSTERS
def inner_vertices_order(vertex_matrix, route, C):
    verticies = {}

    for r in range(1, len(route)):
        prev_c = route[r-1]
        next_c = route[r]
        next_v = vertex_matrix[prev_c-1][next_c-1]
        prev_v = [next_v[0], next_v[1]]
        next_v = [next_v[2], next_v[3]]

        if prev_c not in verticies:
            verticies[prev_c] = []
        verticies[prev_c].append(prev_v)

        if next_c not in verticies:
            verticies[next_c] = []
        verticies[next_c].append(next_v)

    return verticies

# HELPER FUNCTION
def min_dist(coords1, coords2):
    min_d = float('inf')
    vertices = []
    for i in range(len(coords1)):
        pos1 = coords1[i]
        x1 = pos1[0]
        y1 = pos1[1]
        for j in range(len(coords2)):
            pos2 = coords2[j]
            x2 = pos2[0]
            y2 = pos2[1]
            d = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
            if (d < min_d):
                min_d = d
                vertices = [x1, y1, x2, y2]
    return [min_d, vertices]

################## ORDERING CLUSTERS - MINIMIZING DISTANCE 2 ##################
def shortest_cluster_path_2(c, vertex_matrix, cost_matrix):
    #generate all potential paths
    all_perm_paths = []
    first_list = []
    for cluster in c:
        first_list.append(cluster)
    all_perm_paths = list(itertools.permutations(first_list))

    #find shortest path
    min_dist = 10000
    vertices_path = []
    path = []
    for i in range(len(all_perm_paths)):
        cur_path = all_perm_paths[i]
        cur_dist = distance_of_path(cur_path, cost_matrix)
        if (cur_dist < min_dist):
            min_dist = cur_dist
            path = cur_path
            vertices_path = inner_vertices_order(vertex_matrix, path, c)

    #return - (1) path, (2) distance, (3) vertices path
    return [path, min_dist, vertices_path]

def distance_of_path(path, cost_matrix):
    d = 0
    for i in range(1, len(path)):
        p_c = path[i-1]
        n_c = path[i]
        cost = cost_matrix[i-1][i]
        d += cost
    return d

###############################################################################
###############################################################################
################ ORDERING CLUSTERS - PRIORITIZE LARGE MARKINGS ################

# ORDER CLUSTERS
def path_prioritize_largest_clusters(C):
    list_areas = []
    for cluster in C:
        coords = C[cluster]
        num_points = len(coords)
        l = [cluster, num_points]
        list_areas.append(l)

    list_areas.sort(key=lambda x:x[1], reverse=True)

    #print("Density list:")
    #print(list_areas)
    #print()

    path = []
    for i in range(len(list_areas)):
        path.append(list_areas[i][0])

    return path

# FIND VERTICES ORDERING
def vertices_path_prioritize_largest_clusters(path, vertex_matrix):
    v_path = {}

    for i in range(1, len(path)):
        p_c = path[i-1]
        n_c = path[i]
        connecting_coords = vertex_matrix[p_c-1][n_c-1]
        p_c_end = [connecting_coords[0], connecting_coords[1]]
        n_c_start = [connecting_coords[2], connecting_coords[3]]
        try:
            v_path[p_c].append(p_c_end)
        except:
            v_path[p_c] = [p_c_end]
        try:
            v_path[n_c].append(n_c_start)
        except:
            v_path[n_c] = [n_c_start]

    return v_path

###############################################################################
###############################################################################
################################ CLUSTER PATHS ################################

# MAIN FOR CLUSTER PATHS
def new_order_and_shortest_distance(vertices, C, path, n):
    all_paths_and_distances = {}

    for i in range(len(path)):
        cluster = path[i]
        coords = C[cluster]
        #print("Cluster " + str(cluster))
        start = []
        end = []
        v = vertices[cluster]
        bool_start = True
        if (len(v) == 1 and i == 0): #start
            end = v[0]
            start = closest_to_zero(coords, end)
            bool_start = False
        elif (len(v) == 1): #last
            start = v[0]
            end = [n,0]
            #end = coords(len(coords)-1)
        else:
            start = v[0]
            end = v[1]

        new_coords = []
        for j in range(len(coords)):
            if coords[j] != start and coords[j] != end:
                new_coords.append(coords[j])
        len_c = len(new_coords)
        p_list = []
        max_dist = len_c*2
        heapPermutation(new_coords, len_c, len_c, max_dist, p_list)

        if (len(p_list) == 0):
            max_dist = len_c*2
            heapPermutation(new_coords, len_c, len_c, max_dist, p_list)

        for j in range(len(p_list)):
            p_list[j].insert(0,start)
            p_list[j].append(end)
            #if (i == 0):
            #    del p_list[j][0]
            #    p_list[j].insert(0,start)

        OPT = find_shortest_path(p_list)
        opt_path = OPT[0]
        opt_dist = OPT[1]

        #print("Cluster = " + str(cluster))
        #print("Optimal Path = " + str(opt_path))
        #print("Optimal Distance = " + str(opt_dist))
        #print()
        all_paths_and_distances[cluster] = [opt_path, opt_dist]
    return all_paths_and_distances

# Generating permutation using Heap Algorithm
def heapPermutation(a, size, n, max_dist, all):
    # if size becomes 1 then prints the obtained
    # permutation
    #print("heap ", end = " ")
    if (size == 1):
        all.append(a.copy())
        return
    for i in range(size):
        d = calc_list_distance(a)
        if (d < max_dist):
            heapPermutation(a,size-1,n,max_dist, all);
        # if size is odd, swap first and last
        # element
        # else If size is even, swap ith and last element
            if size&1:
                a[0], a[size-1] = a[size-1],a[0]
            else:
                a[i], a[size-1] = a[size-1],a[i]

def find_shortest_path(p_list):
    shortest_path = []
    min_dist = 100000000
    for i in range(len(p_list)):
        cur_dist = calc_list_distance(p_list[i])
        if (cur_dist < min_dist):
            min_dist = cur_dist
            shortest_path = p_list[i]
    op = [shortest_path, min_dist]
    return op

def calc_list_distance(coords):
    dist = 0
    for i in range(1, len(coords)):
        x0 = coords[i-1][0]
        y0 = coords[i-1][1]
        x1 = coords[i][0]
        y1 = coords[i][1]
        dist += math.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))
    return dist

def closest_to_zero(coords, end):
    start = []
    min_dist = 10000
    for i in range(len(coords)):
        if (coords[i] != end):
            x = coords[i][0]
            y = coords[i][1]
            d = math.sqrt(x*x + y*y)
            if (d < min_dist):
                min_dist = d
                start = [x,y]
    return start

###############################################################################
###############################################################################
############################### PRINT FUNCTIONS ###############################
def print_grid(G):
    for i in range(len(G)):
        for j in range(len(G[0])):
            print(int(G[i][j]), end=' ')
        print()

def print_dict(d):
    for key in d:
        print(str(key) + ":" + str(d[key]))
        print()

def print_verticies(G):
    for i in range(len(G)):
        for j in range(len(G[i])):
            print(G[i][j], end = ' ')
        print()

def convert_to_ratio(path, l_g, w_g):
    new_path = []
    w_ratio = w_b / w_g
    l_ratio = l_b / l_g
    #print("w_ratio = " + str(w_ratio))
    #print("l_ratio = " + str(l_ratio))
    for p in range(len(path)):
        coord = path[p]
        x = coord[0]*w_ratio
        y = coord[1]*l_ratio
        new_coord = [x,y]
        new_path.append(new_coord)
    return new_path

def convert_to_cartesian(path):
    L = round(l_b / l_e) #no of lengthwise chunks
    W = round(w_b / w_e) #no of widthwise chunks
    pt_W = w_b / W
    pt_L = l_b / L

    new_path = []
    ax = []
    ay = []
    for p in range(len(path)-1):
        coord = path[p]
        p_x = float(coord[0])
        p_y = float(coord[1])
        c_x = p_y + pt_W/2
        c_y = l_b - p_x - pt_L/2
        new_coord = [c_x, c_y]
        new_path.append(new_coord)
        ax.append(c_x)
        ay.append(c_y)

    return [ax, ay]

def image_padding(img, best_down, best_right):
    width, height = img.size
    L = round(l_b / l_e) #no of lengthwise chunks
    W = round(w_b / w_e) #no of widthwise chunks
    p_L = round(height / L) #pixels in each lengthwise chunk
    p_W = round(width / W) #pixels in each widthwise chunk

    border = (p_W-best_right, p_L-best_down, best_right, best_down)
    bimg = ImageOps.expand(img, border = border, fill = "white")

    return bimg

def lines_on_pic2(image):
    height = 11
    width = 22

    img_width, img_length = image.size
    ig = image

    w_ratio = round(img_width / width)
    l_ratio = round(img_length / height)

    j = 0
    count = 0
    for i in range(img_width):
        if (j == w_ratio):
            shape = [(i,0), (i, img_length)]
            img1 = ImageDraw.Draw(ig)
            img1.line(shape, fill ="blue", width = 1)
            j = 0
            count = count + 1
        j = j + 1

    j = 0
    count = 0
    for i in range(img_length):
        if (j == l_ratio):
            shape = [(0,i), (img_width, i)]
            img1 = ImageDraw.Draw(ig)
            img1.line(shape, fill ="blue", width = 1)
            j = 0
            count = count + 1
        j = j + 1

def graph_points(folder, img, optimization, best_down, best_right, x, y):
    if (optimization == "distance with shuffled grid"):
        title = "Path with shuffled grid"
        img_title = "shuffled_grid.png"
    if (optimization == "distance"):
        title = "Path with original grid"
        img_title = "original_grid.png"
    if (optimization == "largest clusters, noshuf"):
        title = "Path with prioritizing largest markings, original grid"
        img_title = "largest_markings_original_grid.png"
    if (optimization == "largest clusters, shuf"):
        title = "Path with prioritizing largest markings, shuffled grid"
        img_title = "largest_markings_shuffled_grid.png"

    if (optimization == "distance" or optimization == "largest clusters, noshuf"):
        img_padded = img
        #newimg = lines_on_pic2(img_padded)
        img_path = folder + "/" + "original_image_simgrid.png"
        img_padded.save(img_path)
    else:
        img_padded = image_padding(img, best_down, best_right)
        #newimg = lines_on_pic2(img_padded)
        img_path = folder + "/" + "image_for_sim_grid.png"
        img_padded.save(img_path)

    img_g = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img_g, extent=[0, w_b, 0, l_b])
    plt.scatter(x, y)
    path = folder + "/" + img_title
    path = str(path)
    plt.savefig(path)

def distance_travelled(x, y):
    d = 0
    for i in range(1,len(x)):
        x1 = x[i-1]
        y1 = y[i-1]
        x2 = x[i]
        y2 = y[i]
        di = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
        d = d + di
    return d

###############################################################################

def main_helper(folder, img_name, optimization):
    start = time.time()

    img_path = str(folder + "/" + img_name)
    img = Image.open(img_path) #open image
    g = fill_grid(img) #baseline grid
    if (optimization == "distance with shuffled grid" or optimization == "largest clusters, shuf"):
        R = shuffle_grid(g, img) #smaller grid
        g = R[0]
        best_down = R[1]
        best_right = R[2]
        #print("Move down = " + str(best_down))
        #print("Move right = " + str(best_right))
        #print()
    if (optimization == "distance" or optimization == "largest clusters, noshuf"):
        g = compartmentalize_grid(g)
        #print_grid(g)
        best_down = 0
        best_right = 0
    #print_grid(g)
    #print()
    S = separate_clusters(g) #create initial clusters
    c = S[0]
    c = clean_clusters(c)
    c = combine_clusters(c, .75) #clusters are done, c = {cluster #: list of coordinates}
    g = create_nummed_grid(c, g) #display grid with labeled clusters
    D = density_of_clusters(img, c) #density of markings in each cluster
    #print_grid(g)
    #print()
    #print("C = " + str(c))
    num_c = len(c)
    t = time.time() - start
    st = time.time()
    #print("Time elapsed to create grid and clusters = " + str(t))
    # CLUSTER PATH - TSP TO MINIMIZE DISTANCE
    if (num_c > 1):
        W = create_weighted_graph(c)
        cost_matrix = W[0] #
        vertex_matrix = W[1]

        if (optimization == "distance" or optimization == "distance with shuffled grid"):
            X = shortest_cluster_path_2(c, vertex_matrix, cost_matrix)
            path = X[0]
            dist = X[1]
            inner_v = X[2]

        elif (optimization == "largest clusters, noshuf" or optimization == "largest clusters, shuf"):
            path = path_prioritize_largest_clusters(c)
            inner_v = vertices_path_prioritize_largest_clusters(path, vertex_matrix)

    else:
        path = [1]
        inner_v = {1: [[0, 0], [len(g), 0]]}

    #print("Path = " + str(path))
    #print()
    #print("Vertices path = " + str(inner_v))
    #print()
    t = time.time() - st
    st = time.time()
    #print("Time elapsed for all cluster paths = " + str(t))
    #print()

    # CLUSTER PATH - PRIORITIZE BIGGET CLUSTERS
    #PATH FOR EACH CLUSTER
    #print("pre cluster path")
    all_cluster_paths_dict = new_order_and_shortest_distance(inner_v, c, path, len(g))
    #print("post cluster path")
    total_path = []
    total_distance = 0
    for cluster in all_cluster_paths_dict:
        path = all_cluster_paths_dict[cluster][0]
        distance = all_cluster_paths_dict[cluster][1]
        for p in range(len(path)):
            total_path.append(path[p])
        total_distance += distance
    t = time.time() - st
    st = time.time()
    #print("Time elapsed for each cluster = " + str(t))
    #PRINT COMPLETE PATH AND DISTANCES
    #print()
    #print("Total path: ")
    #print(total_path)
    #print()
    #print("Total distance: ")
    #print(total_distance)
    #print()

    #PRINT GRID DIMENSIONS
    w_g = len(g[0])
    l_g = len(g)

    #CARTESIAN PATH
    p1 = convert_to_ratio(total_path, l_g, w_g)
    p2 = convert_to_cartesian(p1)
    x = p2[0]
    y = p2[1]
    #print("Cartesian Path:")
    #print("Ax:")
    #print(x)
    #print("Ay:")
    #print(y)
    #print()

    #GRAPH OF IMAGE WITH POINTS
    graph_points(folder, img, optimization, best_down, best_right, x, y)

    print("Total time elapsed = " + str(time.time() - start))
    #print("Total distance traveled = " + str(distance_travelled(x,y)))

def main():
    folder = "10_equations2"
    img_name = "z_original_image_simgrid.png"
    optimizations = ["distance", "distance with shuffled grid", "largest clusters, noshuf", "largest clusters, shuf"]
    for o in range(len(optimizations)):
        opt = optimizations[o]
        if (o == 0):
            print("ORIGINAL GRID")
        if (o == 1):
            print("SHUFFLED GRID")
        if (o == 2):
            print("LARGEST CLUSTERS, ORIGINAL GRID")
        if (o == 3):
            print("LARGEST CLUSTERS, SHUFFLED GRID")
        print()
        main_helper(folder, img_name, opt)
        print()

if __name__ == "__main__":
    main()
