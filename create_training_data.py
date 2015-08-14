import os
import numpy as np
from multiprocessing import Process

def sgf_to_moves_list(filename):
    f = open(filename, 'r')
    content = f.read()
    f.close()

    if 'AB[' in content or 'AW[' in content:
        return False

    moves = []
    ind = 0

    while True:
        if len(moves) % 2 == 0:
            index = content.find(';B[', ind)
            if index == -1:
                break
        else:
            index = content.find(';W[', ind)
            if index == -1:
                break

        if content[index + 3] == ']':
            moves.append('pass')
            ind = index + 4
            continue

        x, y = [ord(letter) - ord('a') for letter in content[index+3:index+5]]

        moves.append((x, y))
        ind = index + 6

    return moves

in_bounds = lambda x, y: 0 <= x < 19 and 0 <= y < 19

def print_small_board(board, illegal):
    for i in range(19):
        text = ""
        for j in range(19):
            text += ('o' if illegal[i][j] else '-') if not (0 <= board[i][j] <= 1) else ('B' if board[i][j] == 0 else 'W')
        print text

def update_board(board, group_map, groups, empty_groups, illegal, move, player):
    '''
        board: board[y][x] == 0 if (x, y) is black, 1 if (x, y) is white, -1 if empty
        group_map: group_map[y][x] == group_index of (x, y), -1 if (x, y) is empty
        groups: list of groups indexed by group_index
                Each group contains (liberties, pieces)
                liberites is a set of all (x, y) liberties of the group
                pieces is a set of all stones in the group
                elements can be None if group has been removed
        empty_groups: list of indices of all empty groups
        illegal: illegal[y][x] == 1 if move is illegal, 0 otherwise
        move: (x, y) of move
        player: 0 if black is moving, 1 if white is moving
    '''

    x, y = move
    opponent = 1 - player
    board[y][x] = player

    liberties = set()
    allies = set()
    enemies = set()
    for x2, y2 in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
        if in_bounds(x2, y2):
            if board[y2][x2] == -1:
                liberties.add((x2, y2))
            elif board[y2][x2] == player:
                allies.add(group_map[y2][x2])
            else:
                enemies.add(group_map[y2][x2])

    if len(allies) == 0:
        new_group = (liberties, set([move]))
        if empty_groups:
            group_ind = empty_groups.pop()
            groups[group_ind] = new_group
        else:
            groups.append(new_group)
            group_ind = len(groups) - 1

        group_map[y][x] = group_ind

    else:
        group_ind = min(allies)
        group_map[y][x] = group_ind

        new_liberties, new_pieces = groups[group_ind]
        for other_index in allies:
            if other_index == group_ind:
                continue
            other_liberties, other_pieces = groups[other_index]
            new_liberties.update(other_liberties)
            new_pieces.update(other_pieces)
            temp = groups[other_index]
            for x2, y2 in other_pieces:
                group_map[y2][x2] = group_ind

            groups[other_index] = None
            empty_groups.append(other_index)

        new_liberties.remove((x, y))

        new_pieces.add((x, y))
        new_liberties.update(liberties)

    captured = []
    for enemy in enemies:
        groups[enemy][0].remove((x, y))
        if len(groups[enemy][0]) == 0:
            pieces = groups[enemy][1]

            captured.extend(pieces)
            for x2, y2 in pieces:
                board[y2][x2] = -1
                group_map[y2][x2] = -1

                for x3, y3 in [(x2-1, y2), (x2+1, y2), (x2, y2-1), (x2, y2+1)]:
                    if in_bounds(x3, y3) and board[y3][x3] == player:
                        groups[group_map[y3][x3]][0].add((x2, y2))

            groups[enemy] = None
            empty_groups.append(enemy)

    for x2 in range(19):
        for y2 in range(19):
            if board[y2][x2] != -1:
                illegal[y2][x2] = True
                continue

            is_illegal = True
            for x3, y3 in [(x2-1, y2), (x2+1, y2), (x2, y2-1), (x2, y2+1)]:
                if not in_bounds(x3, y3):
                    continue
                if board[y3][x3] == -1:
                    is_illegal = False
                    break
                if board[y3][x3] == player:
                    if len(groups[group_map[y3][x3]][0]) == 1:
                        is_illegal = False
                        break
                    continue

                if len(groups[group_map[y3][x3]][0]) >= 1:
                    is_illegal = False
                    break

            if is_illegal:
                illegal[y2][x2] = True
                continue

    if len(captured) == 1:
        x2, y2 = captured[0]

        is_ko = True
        for x3, y3 in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if x2 == x3 and y2 == y3:
                continue
            if not in_bounds(x3, y3):
                continue
            if board[y3][x3] == player or board[y3][x3] == -1:
                is_ko = False

        if is_ko:
            for x3, y3 in [(x2-1, y2), (x2+1, y2), (x2, y2-1), (x2, y2+1)]:
                if x2 == x3 and y2 == y3:
                    continue
                if not in_bounds(x3, y3):
                    continue
                if board[y3][x3] == opponent or board[y3][x3] == -1:
                    is_ko = False

            if is_ko:
                illegal[y2][x2] = True

edge_array = [[1 for _ in range(19)] for __ in range(19)]

def add_board_to_array(board, group_map, groups, illegal, player, inputs, ind):
    '''
    adds board to array
    '''
    opponent = 1 - player

    # Each array contains 3 19x19 boards - holding groups with 1, 2, >2 liberties respectively. 
    player_array = [[[0 for _ in range(19)] for __ in range(19)] for ___ in range(3)]
    opponent_array = [[[0 for _ in range(19)] for __ in range(19)] for ___ in range(3)]
    illegal_array = [[0 for _ in range(19)] for __ in range(19)]

    for x in range(19):
        for y in range(19):
            if illegal[y][x]:
                illegal_array[y][x] = 1

            stone = board[y][x]

            if stone == -1:
                continue

            num_liberties = len(groups[group_map[y][x]][0])
            liberty_index = 0 if num_liberties == 1 else (1 if num_liberties == 2 else 2)

            if stone == player:
                player_array[liberty_index][y][x] = 1
            else:
                opponent_array[liberty_index][y][x] = 1

    inputs[ind][0] = player_array[0]
    inputs[ind][1] = player_array[1]
    inputs[ind][2] = player_array[2]
    inputs[ind][3] = opponent_array[0]
    inputs[ind][4] = opponent_array[1]
    inputs[ind][5] = opponent_array[2]
    inputs[ind][6] = illegal_array
    inputs[ind][7] = edge_array

def add_move_to_array(move, targets, ind):
    # adds move to target array
    targets[ind] = move[1]*19 + move[0]

def sgf_to_numpy(filename):
    moves = sgf_to_moves_list(filename)
    if not moves:
        return False
    board = [[-1 for _ in range(19)] for __ in range(19)]
    group_map = [[-1 for _ in range(19)] for __ in range(19)]
    groups = []
    illegal = [[False for _ in range(19)] for __ in range(19)]
    empty_groups = []

    num_nonpass_moves = len(moves) - moves.count('pass')
    inputs = [[None for _ in range(8)] for __ in range(num_nonpass_moves)]
    targets = [None for _ in range(num_nonpass_moves)]
    ind = 0
    for i, move in enumerate(moves):
        if move == 'pass':
            continue
        player = i % 2

        add_board_to_array(board, group_map, groups, illegal, player, inputs, ind)

        add_move_to_array(move, targets, ind)
        illegal = [[False for _ in range(19)] for __ in range(19)]
        update_board(board, group_map, groups, empty_groups, illegal, move, player)

        ind += 1

    return (np.array(inputs), np.array(targets))

sgf_folder = '/Users/dzd123/Documents/Summer 2015/GoBot/datasets/'
output_folder = '/Users/dzd123/Documents/Summer 2015/GoBot/training_data/'

def read_files(index, files):
    print len(files)
    for i, filename in enumerate(files):
        result = sgf_to_numpy(sgf_folder+filename)
        if not result:
            continue
        inputs, targets = result
        np.savez_compressed(output_folder+filename[:-4], inputs=inputs, targets=targets)
        if i % 100 == 0:
            print index, i

def read_sgf_folder(num_processes=4):
    files = os.listdir(sgf_folder)
    num_files = len(files)
    files_per_process = num_files/num_processes

    processes = []
    for i in range(num_processes-1):
        p = Process(target=read_files, args=(i, files[files_per_process*i : files_per_process*(i+1)]))
        p.start()
        processes.append(p)

    p = Process(target=read_files, args=(num_processes-1, files[files_per_process*(num_processes-1):]))
    p.start()
    processes.append(p)

    for process in processes:
        process.join()

if __name__ == '__main__':

    read_sgf_folder()
