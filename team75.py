import sys
import random
import signal
import time
import copy
import math

class Player75():


    def __init__(self):
        self.deepcopy = copy.deepcopy
        self.hor = [ 5, 6, 7, 8]
        self.ver = [ 9, 10, 11, 12]
        self.dia = [ 13, 14]
        self.my_constants = [ 0, 5, 40, 200, 3000 ]
        self.his_constants = [ 0, 5, 40, 200, 3000 ]
        self.max_level = 26

    def print_board(self):
		# for printing the state of the board
		print '==============Board State=============='
		for i in range(16):
			if i%4 == 0:
				print
			for j in range(16):
				if j%4 == 0:
					print "",
				print self.board_status[i][j],
			print
		print

		print '==============Block State=============='
		for i in range(4):
			for j in range(4):
				print self.block_status[i][j],
			print
		print '======================================='
		print
		print

    def print_utility(self, flag, revflag):
        mup = self.my_utility_params
        hup = self.his_utility_params
        mu = self.my_utility
        hu = self.his_utility

        for bi in range(4):
            for bj in range(4):
                print "----------------------"
                print bi, bj
                print "----------------------"
                print "my", flag
                for k in range(15):
                    if k % 4 == 1 and k != 1:
                        print
                    print mup[bi][bj][k],
                print
                print "-------", mu[bi][bj],"----------"
                print "his", revflag
                for k in range(15):
                    if k % 4 == 1 and k != 1:
                        print
                    print hup[bi][bj][k],
                print
                print "-------", hu[bi][bj],"----------"
                print

    def print_board_utility(self):
        for k in range(15):
            if k % 4 == 1 and k != 1:
                print
            print self.my_board_utility_params_dc[k],
        print
        print self.my_board_utility_dc
        print
        for i in range(4):
            for j in range(4):
                print self.my_utility_dc[i][j],
            print
        print

        for k in range(15):
            if k % 4 == 1 and k != 1:
                print
            print self.his_board_utility_params_dc[k],
        print
        print self.his_board_utility_dc
        print
        for i in range(4):
            for j in range(4):
                print self.his_utility_dc[i][j],
            print
        print

        for i in range(4):
            for j in range(4):
                print self.my_utility_dc[i][j],
            print
        print
        print

    def get_available_moves(self):
        self.available_moves = [[[] for j in range(4)] for i in range(4)]
        for i in range(16):
            for j in range(16):
                if self.board_status[i][j] == '-' and self.block_status[i/4][j/4] == '-':
                    self.available_moves[i/4][j/4].append((i,j))
        return

    def init_utility(self, flag, revflag):
        self.my_utility_params_dc = [ [ [ 0 for k in range(15) ] for j in range(4)] for i in range(4)]
        self.his_utility_params_dc = [ [ [ 0 for k in range(15) ] for j in range(4)] for i in range(4)]
        self.my_utility_dc = [ [ 0 for j in range(4) ] for i in range(4)]
        self.his_utility_dc = [ [ 0 for j in range(4) ] for i in range(4)]

        self.my_board_utility_params_dc = [ 0 for k in range(15) ]
        self.his_board_utility_params_dc = [ 0 for k in range(15) ]
        self.my_board_utility_dc = 0
        self.his_board_utility_dc = 0

        bs = self.deep_copied_board_status
        bos = self.deep_copied_block_status
        mup = self.my_utility_params_dc
        hup = self.his_utility_params_dc
        mu = self.my_utility_dc
        hu = self.his_utility_dc
        mbup = self.my_board_utility_params_dc
        hbup = self.his_board_utility_params_dc

        hor = self.hor
        ver = self.ver
        dia = self.dia
        my_constants = self.my_constants
        his_constants = self.his_constants


        for i in range(4):
            for j in range(4):
                x = 4*i
                y = 4*j
                if self.deep_copied_block_status[i][j] != '-':
                    mu[i][j] = 1000
                    hu[i][j] = 1000
                for a in range(4):
                    for b in range(4):

                        if bs[x+a][y+b] == flag:
                            mup[i][j][hor[a]] += 1
                            mup[i][j][ver[b]] += 1

                            if a == b:
                                mup[i][j][dia[0]] += 1
                            elif a+b == 3:
                                mup[i][j][dia[1]] += 1

                        elif bs[x+a][y+b] == revflag:
                            hup[i][j][hor[a]] += 1
                            hup[i][j][ver[b]] += 1

                            if a == b:
                                hup[i][j][dia[0]] += 1
                            elif a+b == 3:
                                hup[i][j][dia[1]] += 1

                for k in range(5, 15):
                    if not hup[i][j][k]:
                        mup[i][j][ mup[i][j][k] ] += 1
                    if not mup[i][j][k]:
                        hup[i][j][ hup[i][j][k] ] += 1


                mu[i][j] = my_constants[0] * mup[i][j][0] + my_constants[1] * mup[i][j][1] + my_constants[2] * mup[i][j][2] + my_constants[3] * mup[i][j][3] + my_constants[4] * mup[i][j][4]
                hu[i][j] = his_constants[0] * hup[i][j][0] + his_constants[1] * hup[i][j][1] + his_constants[2] * hup[i][j][2] + his_constants[3] * hup[i][j][3] + his_constants[4] * hup[i][j][4]

        for i in range(4):
            for j in range(4):
                if bos[i][j] == flag:
                    mbup[hor[i]] += 1
                    mbup[ver[j]] += 1

                    if i == j:
                        mbup[dia[0]] += 1
                    elif i+j == 3:
                        mbup[dia[1]] += 1

                elif bos[i][j] == revflag:
                    hbup[hor[i]] += 1
                    hbup[ver[j]] += 1

                    if i == j:
                        hbup[dia[0]] += 1
                    elif i+j == 3:
                        hbup[dia[1]] += 1

        for k in range(5, 15):
            if not hbup[k]:
                mbup[ mbup[k] ] += 1
            if not mbup[k]:
                hbup[ hbup[k] ] += 1


        self.my_board_utility_dc = my_constants[0] * mbup[0] + my_constants[1] * mbup[1] + my_constants[2] * mbup[2] + my_constants[3] * mbup[3] + my_constants[4] * mbup[4]
        self.his_board_utility_dc = his_constants[0] * hbup[0] + his_constants[1] * hbup[1] + his_constants[2] * hbup[2] + his_constants[3] * hbup[3] + his_constants[4] * hbup[4]


    def find_valid_move_cells(self, old_move):
        allowed_cells = []
        allowed_block = [old_move[0]%4, old_move[1]%4]

        if old_move != (-1,-1) and self.block_status[allowed_block[0]][allowed_block[1]] == '-':
            for i in range(4*allowed_block[0], 4*allowed_block[0]+4):
                for j in range(4*allowed_block[1], 4*allowed_block[1]+4):
                    if self.board_status[i][j] == '-':
                        allowed_cells.append((i,j))
        else:
            for i in range(16):
                for j in range(16):
                    if self.board_status[i][j] == '-' and self.block_status[i/4][j/4] == '-':
                        allowed_cells.append((i,j))
        return allowed_cells

    def update_board_utility(self, mbup, hbup, flag, par):
        if not hbup[par]:
            if flag == self.my_main_flag:
                self.my_board_utility += self.my_constants[mbup[par] + 1] - self.my_constants[mbup[par]]
            else:
                self.his_board_utility += self.his_constants[mbup[par] + 1] - self.his_constants[mbup[par]]
        if not mbup[par]:
            if flag == self.my_main_flag:
                self.his_board_utility -= self.his_constants[hbup[par]]
            else:
                self.my_board_utility -= self.my_constants[hbup[par]]

    def update_board_utility_tie(self, mbup, hbup, flag, par):
        if flag == self.my_main_flag:
            self.my_board_utility -= self.my_constants[mbup[par]]
            self.his_board_utility -= self.his_constants[hbup[par]]
        else:
            self.my_board_utility -= self.my_constants[hbup[par]]
            self.his_board_utility -= self.his_constants[mbup[par]]


    def update_utility(self, move, flag):
        x = move[0]/4
        y = move[1]/4
        i = move[0]%4
        j = move[1]%4
        hor = self.hor
        ver = self.ver
        dia = self.dia
        if flag == self.my_main_flag:
            mup = self.my_utility_params
            hup = self.his_utility_params
            mu = self.my_utility
            hu = self.his_utility
            mbup = self.my_board_utility_params_dc
            hbup = self.his_board_utility_params_dc
            my_constants = self.my_constants
            his_constants = self.his_constants
        else:
            hup = self.my_utility_params
            mup = self.his_utility_params
            hu = self.my_utility
            mu = self.his_utility
            hbup = self.my_board_utility_params_dc
            mbup = self.his_board_utility_params_dc
            his_constants = self.my_constants
            my_constants = self.his_constants

        if self.block_status[x][y] != '-':
            self.my_utility[x][y] = 1000
            self.his_utility[x][y] = 1000
            if self.block_status[x][y] == flag:
                self.update_board_utility(mbup, hbup, flag, hor[x])
                self.update_board_utility(mbup, hbup, flag, ver[y])
                if x==y:
                    self.update_board_utility(mbup, hbup, flag, dia[0])
                elif x+y==3:
                    self.update_board_utility(mbup, hbup, flag, dia[1])
            elif self.block_status[x][y] == 'd':
                self.update_board_utility_tie(mbup, hbup, flag, hor[x])
                self.update_board_utility_tie(mbup, hbup, flag, ver[y])
                if x==y:
                    self.update_board_utility_tie(mbup, hbup, flag, dia[0])
                elif x+y==3:
                    self.update_board_utility_tie(mbup, hbup, flag, dia[1])
            else:
                pass
            return


        if not hup[x][y][hor[i]]:
            mu[x][y] += my_constants[mup[x][y][hor[i]] + 1] - my_constants[mup[x][y][hor[i]]]
        if not mup[x][y][hor[i]]:
            hu[x][y] -= his_constants[hup[x][y][hor[i]]]
        mup[x][y][hor[i]] += 1

        if not hup[x][y][ver[j]]:
            mu[x][y] += my_constants[mup[x][y][ver[j]] + 1] - my_constants[mup[x][y][ver[j]]]
        if not mup[x][y][ver[j]]:
            hu[x][y] -= his_constants[hup[x][y][ver[j]]]
        mup[x][y][ver[j]] += 1

        if i==j:
            if not hup[x][y][dia[0]]:
                mu[x][y] += my_constants[mup[x][y][dia[0]] + 1] - my_constants[mup[x][y][dia[0]]]
            if not mup[x][y][dia[0]]:
                hu[x][y] -= his_constants[hup[x][y][dia[0]]]
            mup[x][y][dia[0]] += 1
        elif i+j==3:
            if not hup[x][y][dia[1]]:
                mu[x][y] += my_constants[mup[x][y][dia[1]] + 1] - my_constants[mup[x][y][dia[1]]]
            if not mup[x][y][dia[1]]:
                hu[x][y] -= his_constants[hup[x][y][dia[1]]]
            mup[x][y][dia[1]] += 1




    def get_move(self, old_move, flag, play_random):
        allowed_cells = []
        allowed_block = [old_move[0]%4, old_move[1]%4]

        if old_move != (-1,-1) and self.block_status[allowed_block[0]][allowed_block[1]] == '-':
            allowed_cells.extend(self.available_moves[allowed_block[0]][allowed_block[1]])
        else:
            for i in range(4):
                for j in range(4):
                    allowed_cells.extend(self.available_moves[i][j])

        if play_random:
            return allowed_cells[ random.randint( 0, len(allowed_cells)-1 ) ]

        # choose according to utility
        #---------------------------------------

        utility_arr = [(0,0) for i in xrange(len(allowed_cells))]
        hor = self.hor
        ver = self.ver
        dia = self.dia

        if flag == self.my_main_flag:
            mup = self.my_utility_params
            hup = self.his_utility_params
            mu = self.my_utility
            hu = self.his_utility
            mbup = self.my_board_utility_params
            hbup = self.his_board_utility_params
            my_constants = self.my_constants
            his_constants = self.his_constants
            mbu = self.my_board_utility
            hbu = self.his_board_utility
        else:
            hup = self.my_utility_params
            mup = self.his_utility_params
            hu = self.my_utility
            mu = self.his_utility
            hbup = self.my_board_utility_params
            mbup = self.his_board_utility_params
            his_constants = self.my_constants
            my_constants = self.his_constants
            hbu = self.my_board_utility
            mbu = self.his_board_utility

        count = 0
        #total = 0
        #mi = 1000000000
        for i, j in allowed_cells:
            x = i/4
            y = j/4
            i = i%4
            j = j%4
            my_ut = mu[x][y]
            his_ut = hu[x][y]

            if not hup[x][y][hor[i]]:
                my_ut += my_constants[mup[x][y][hor[i]] + 1] - my_constants[mup[x][y][hor[i]]]
            if not hup[x][y][ver[j]]:
                my_ut += my_constants[mup[x][y][ver[j]] + 1] - my_constants[mup[x][y][ver[j]]]

            if not mup[x][y][hor[i]]:
                his_ut -= his_constants[hup[x][y][hor[i]]]
            if not mup[x][y][ver[j]]:
                his_ut -= his_constants[hup[x][y][ver[j]]]

            if i==j:
                if not hup[x][y][dia[0]]:
                    my_ut += my_constants[mup[x][y][dia[0]] + 1] - my_constants[mup[x][y][dia[0]]]
                if not mup[x][y][dia[0]]:
                    his_ut -= his_constants[hup[x][y][dia[0]]]
            if i+j==3:
                if not hup[x][y][dia[1]]:
                    my_ut += my_constants[mup[x][y][dia[1]] + 1] - my_constants[mup[x][y][dia[1]]]
                if not mup[x][y][dia[1]]:
                    his_ut -= his_constants[hup[x][y][dia[1]]]

            if self.block_status[i][j] != '-':
                final_ut = 30 * (my_ut - mu[x][y]) - 80 * hbu
            else:
                c = 1
                if mbup[hor[i]]:
                    c += his_constants[hbup[hor[i]]]
                if mbup[ver[j]]:
                    c += his_constants[hbup[ver[j]]]
                if i==j:
                    c += his_constants[hbup[dia[0]]]
                elif i+j==3:
                    c += his_constants[hbup[dia[1]]]
                final_ut = 30 * (my_ut - mu[x][y]) + 5*(hu[x][y] - his_ut) - 7 * int(math.sqrt(c * hu[i][j]))

            #if final_ut < mi:
            #    mi = final_ut
            utility_arr[count] = (final_ut, count)
            #total += final_ut
            count += 1
        le = len(allowed_cells)
        """
        rand_number = random.randint(0, total - mi*le)
        total = 0
        count = 0
        for u in utility_arr:
            total += u - mi
            if total >= rand_number:
                break
            count += 1
        """
        utility_arr = sorted(utility_arr, reverse=True)
        maxi7 = utility_arr[0][0] * 0.5
        count = 0
        for u in utility_arr:
            if u[0] < maxi7:
                break
            count += 1

        #---------------------------------------
        if count >= le:
            count = le - 1
        #print utility_arr
        #print utility_arr[count]
        return allowed_cells[utility_arr[random.randint(0,count)][1]]



    def play_move(self, new_move, ply):
        self.board_status[new_move[0]][new_move[1]] = ply
        x = new_move[0]/4
        y = new_move[1]/4
        x4 = 4*x
        y4 = 4*y
        self.available_moves[x][y].remove((new_move[0], new_move[1]))
        bs = self.board_status

        for i in xrange(4):
    	    if (bs[x4+i][y4] == bs[x4+i][y4+1] == bs[x4+i][y4+2] == bs[x4+i][y4+3] == ply):
                self.block_status[x][y] = ply
                del self.available_moves[x][y][:]
                self.update_utility(new_move, ply)
                return
            if (bs[x4][y4+i] == bs[x4+1][y4+i] == bs[x4+2][y4+i] == bs[x4+3][y4+i] == ply):
                self.block_status[x][y] = ply
                del self.available_moves[x][y][:]
                self.update_utility(new_move, ply)
                return

        if (bs[x4][y4] == bs[x4+1][y4+1] == bs[x4+2][y4+2] == bs[x4+3][y4+3] == ply):
            self.block_status[x][y] = ply
            del self.available_moves[x][y][:]
            self.update_utility(new_move, ply)
            return
        if (bs[x4+3][y4] == bs[x4+2][y4+1] == bs[x4+1][y4+2] == bs[x4][y4+3] == ply):
            self.block_status[x][y] = ply
            del self.available_moves[x][y][:]
            self.update_utility(new_move, ply)
            return

        for i in xrange(4):
            for j in xrange(4):
                if bs[x4+i][y4+j] =='-':
                    self.update_utility(new_move, ply)
                    return
        self.block_status[x][y] = 'd'
        del self.available_moves[x][y][:]
        self.update_utility(new_move, ply)
        return

    def check_big(self):

        bs = self.block_status
    	for i in xrange(4):
            if (bs[i][0] == bs[i][1] == bs[i][2] == bs[i][3]):
                return bs[i][0]
            if (bs[0][i] == bs[1][i] == bs[2][i] == bs[3][i]):
                return bs[0][i]

        if (bs[0][0] == bs[1][1] == bs[2][2] == bs[3][3]):
            return bs[0][0]
        if (bs[3][0] == bs[2][1] == bs[1][2] == bs[0][3]):
            return bs[3][0]

        for i in xrange(4):
            for j in xrange(4):
                if bs[i][j] == '-':
                    return '-'
        return 'd'


    def play_a_game(self, cell, flag):

        X = cell[0]/4
        Y = cell[1]/4
        x = cell[0]%4
        y = cell[1]%4
        if flag == 'o':
            revflag = 'x'
        else :
            revflag = 'o'

        # play first move and check for termination
        #---------------------------------------
        self.play_move(cell, flag)
        if self.block_status[X][Y] == flag:
            cb = self.check_big()
            if cb != '-':
                return cb
        elif self.block_status[X][Y] == 'd':
            cb = self.check_big()
            if cb == 'd':
                return cb
        #---------------------------------------

        level = 0
        play_random = False
        # main game loop
        #---------------------------------------
        while 1:
            if level > self.max_level:
                play_random = True
            # decide loop's first play
            #---------------------------------------
            now_move = self.get_move((x, y), revflag, play_random)
            X = now_move[0]/4
            Y = now_move[1]/4
            x = now_move[0]%4
            y = now_move[1]%4
            #---------------------------------------

            # play loop's first play
            #---------------------------------------
            self.play_move([4*X+x, 4*Y+y], revflag)
            if self.block_status[X][Y] == revflag:
                cb = self.check_big()
                if cb != '-':
                    return cb
            elif self.block_status[X][Y] == 'd':
                cb = self.check_big()
                if cb == 'd':
                    return cb
            #---------------------------------------

            # decide loop's second play
            #---------------------------------------
            now_move = self.get_move((x, y), flag, play_random)
            X = now_move[0]/4
            Y = now_move[1]/4
            x = now_move[0]%4
            y = now_move[1]%4
            #---------------------------------------

            # play loop's second play
            #---------------------------------------
            self.play_move([4*X+x, 4*Y+y], flag)
            if self.block_status[X][Y] == flag:
                cb = self.check_big()
                if cb != '-':
                    return cb
            elif self.block_status[X][Y] == 'd':
                cb = self.check_big()
                if cb == 'd':
                    return cb
            level += 2
            #---------------------------------------

    def move(self, board, old_move, flag):
        try:
            return self.movemy(board, old_move, flag)
        except:
            cells = board.find_valid_move_cells(old_move)
            return cells[random.randrange(len(cells))]

    def movemy(self, board, old_move, flag):

        cells = board.find_valid_move_cells(old_move)
        self.my_main_flag = flag
        time_per_cell = 1000000 * 13.5 / len(cells)
        wins = 0
        loses = 0
        ties = 0
        start_time = 0
        current_time = 0
        res = 0
        if flag == 'o':
            revflag = 'x'
        else:
            revflag = 'o'

        self.deep_copied_board_status = self.deepcopy(board.board_status)
        self.deep_copied_block_status = self.deepcopy(board.block_status)

        self.init_utility(flag, revflag)
        # self.print_utility(flag, revflag)
        # self.print_board_utility()

        best_prob = -0.01
        best_cell = None

        point_best_prob = -0.01
        point_best_cell = None

        for cell in cells:
            wins = 0
            loses = 0
            ties = 0
            tie_points = 0

            start_time = time.time()*1000000

            while time.time()*1000000 - start_time < time_per_cell:

                self.board_status = self.deepcopy(self.deep_copied_board_status)
                self.block_status = self.deepcopy(self.deep_copied_block_status)
                self.my_utility = self.deepcopy(self.my_utility_dc)
                self.his_utility = self.deepcopy(self.his_utility_dc)
                self.my_utility_params = self.deepcopy(self.my_utility_params_dc)
                self.his_utility_params = self.deepcopy(self.his_utility_params_dc)

                self.my_board_utility_params = self.deepcopy(self.my_board_utility_params_dc)
                self.his_board_utility_params = self.deepcopy(self.his_board_utility_params_dc)
                self.my_board_utility = self.my_board_utility_dc
                self.his_board_utility = self.his_board_utility_dc

                self.get_available_moves()
                res = self.play_a_game(cell, flag)
                if res == flag:
                    wins += 1
                elif res == revflag:
                    loses += 1
                elif res == 'd':
                    ties += 1
                    for i in range(4):
                        for j in range(4):
                            if self.block_status[i][j] == flag:
                                tie_points += 1
                else:
                    pass
            tot = 16*wins + tie_points
            prob = 1.0 * wins / (wins + loses + ties)
            point_prob = 1.0*tot / (wins + loses + ties)

            if prob > best_prob:
                best_prob = prob
                best_cell = cell
            if point_prob > point_best_prob:
                point_best_prob = point_prob
                point_best_cell = cell
            #print cell, prob, best_cell, best_prob
            #print wins, loses, ties, wins+loses+ties
        #print best_prob, best_cell, point_best_prob, point_best_cell

        if best_prob <= 0.05:
            best_cell = point_best_cell

        return best_cell
